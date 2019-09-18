import asyncio
import json
import logging
import os
import tempfile
import time

import aiohttp
import boto3
import datadog
import h5py
from papertasks.events.event import Event

import dataprep2
import settings
from db_worker.corpus_db import CorpusDB

SLEEP_SECONDS = 10
MODEL_VERSION = 2
MAX_NUMBER_OF_MESSAGES = 10  # upper limit for SQS
VISIBILITY_TIMEOUT = 600  # time till messages become visible in queue again, in seconds
POLL_TIMEOUT = 20  # timeout for SQS long polling, in seconds

def _send_all(source, dest, nbytes: int = None):
    nsent = 0
    while nbytes is None or nsent < nbytes:
        tosend = 64 * 1024
        if nbytes is not None:
            tosend = min(tosend, nbytes - nsent)
        buf = source.read(tosend)
        if not buf:
            break
        dest.write(buf)
        nsent += len(buf)
    dest.flush()
    
def parse_event(message):
    try:
        event = Event(**json.loads(message['Body']))
        return event
    except Exception:
        logging.exception('Failed parsing message from queue. Message: %s', str(message))
        return None
    
def delete_batch(queue_client, queue_url, messages_to_delete):
    if messages_to_delete:
        queue_client.delete_message_batch(
            QueueUrl=queue_url,
            Entries=[{
                'Id': message['MessageId'],
                'ReceiptHandle': message['ReceiptHandle']
            } for message in messages_to_delete]
        )

def get_pdf_hash_to_messages(queue_client, queue_url, corpus_db, dd_stats, dd_prefix):
    while True:
        messages = queue_client.receive_message(
            QueueUrl=queue_url,
            VisibilityTimeout=VISIBILITY_TIMEOUT,
            MaxNumberOfMessages=MAX_NUMBER_OF_MESSAGES,
            WaitTimeSeconds=POLL_TIMEOUT
        ).get('Messages', [])

        if len(messages) == 0:
            logging.info("No papers to process. Sleeping for %s", str(SLEEP_SECONDS))
            time.sleep(SLEEP_SECONDS)
            continue

        messages_to_delete = []
        # Need this mapping because of downstream batch processing and we lose paper id to message id
        # correspondence. We need message ids to delete success events from queue.
        pdf_hash_to_messages = dict()
        for m in messages:
            event = parse_event(m)
            if not event or (event.type, event.action) != ('SourcedPaper', 'modified'):
                messages_to_delete.append(m)
            else:
                sha = corpus_db.get_sourced_paper_pdf_hash(event.id)
                if sha:
                    # TODO check ParseStore and skip if (PDF hash, model version, SP version) already exists
                    if sha in pdf_hash_to_messages:
                        pdf_hash_to_messages[sha].append(m)
                    else:
                        pdf_hash_to_messages[sha] = [m]
                else:
                    messages_to_delete.append(m)

        delete_batch(queue_client, queue_url, messages_to_delete)
        num_to_process = len(pdf_hash_to_messages)

        if num_to_process > 0:
            dd_stats.increment(dd_prefix + "attempts", num_to_process)
            return pdf_hash_to_messages

def main():
    if os.name != 'nt':
        import manhole
        manhole.install()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(thread)d %(levelname)s %(message)s', level=logging.INFO)

    dataprep_host = os.environ["DATAPREP_HOST"]
    dataprep_port = int(os.environ["DATAPREP_PORT"])
    logging.info("Dataprep host: %s, port: %d", dataprep_host, dataprep_port)

    corpusdb_kwargs = dict(
        host=os.environ["CORPUS_DB_HOST"],
        user=os.environ["CORPUS_DB_USER"]
    )
    logging.info("Corpus DB config: %s", corpusdb_kwargs)
    corpus_db = CorpusDB(
        password=os.environ["CORPUS_DB_PASSWORD"],
        **corpusdb_kwargs
    )
    env = os.environ["SPV2_ENV"]

    queue_name = os.environ['SQS_QUEUE_NAME']
    queue_client = boto3.client('sqs', region_name='us-west-2')
    queue_url = queue_client.get_queue_url(QueueName=queue_name)['QueueUrl']
    logging.info("SQS queue name: %s", queue_name)

    # start datadog
    datadog.initialize(api_key=os.environ.get("DATADOG_API_KEY"))
    dd_stats = datadog.ThreadStats()
    dd_stats.start()
    dd_prefix = "spv2.%s." % env

    # logging.info("Loading model settings ...")
    # model_settings = settings.default_model_settings
    #
    # logging.info("Loading token statistics ...")
    # token_stats = dataprep2.TokenStatistics("model/all.tokenstats3.gz")
    #
    # logging.info("Loading embeddings ...")
    # embeddings = dataprep2.CombinedEmbeddings(
    #     token_stats,
    #     dataprep2.GloveVectors(model_settings.glove_vectors),
    #     model_settings.embedded_tokens_fraction
    # )
    #
    # import with_labels  # Heavy import, so we do it here
    # model = with_labels.model_with_labels(model_settings, embeddings)
    # model.load_weights("model/C49.h5")

    def featurized_tokens_filenames():
        # async http stuff
        async_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_event_loop)
        connector = aiohttp.TCPConnector(loop=async_event_loop, force_close=True)
        session = aiohttp.ClientSession(connector=connector, read_timeout=120, conn_timeout=120)
        write_lock = asyncio.Lock()
        async def write_json_tokens_to_file(paper_id: str, json_file):
            url = "http://%s:%d/v1/json/paperid/%s" % (dataprep_host, dataprep_port, paper_id)
            attempts_left = 5
            with tempfile.NamedTemporaryFile(prefix="SPv2DBWorker-%s-" % paper_id, suffix=".json") as f:
                f.seek(0)
                f.truncate()
                def write_json_to_output(json_object):
                    f.write(json.dumps(json_object).encode("utf-8"))
                while True:
                    attempts_left -= 1
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                # We write to a tempfile first, because we don't want to end up with
                                # half-written json if something goes wrong while reading from the
                                # socket.
                                while True:
                                    chunk = await response.content.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                dd_stats.increment(dd_prefix + "dataprep.success")
                                break
                            else:
                                dd_stats.increment(dd_prefix + "dataprep.failure")
                                if attempts_left > 0:
                                    logging.error(
                                        "Error %d from dataprep server for paper id %s. %d attempts left.",
                                        response.status,
                                        paper_id,
                                        attempts_left)
                                else:
                                    dd_stats.increment(dd_prefix + "dataprep.gave_up")
                                    logging.error(
                                        "Error %d from dataprep server for paper id %s. Giving up.",
                                        response.status,
                                        paper_id)
                                    error = {
                                        "error": {
                                            "message": "Status %s from dataprep server" % response.status,
                                            "stackTrace": None,
                                            "docName": "%s.pdf" % paper_id
                                        }
                                    }
                                    write_json_to_output(error)
                                    break
                    except Exception as e:
                        dd_stats.increment(dd_prefix + "dataprep.failure")
                        if attempts_left > 0:
                            logging.error(
                                "Error %r from dataprep server for paper id %s. %d attempts left.",
                                e,
                                paper_id,
                                attempts_left)
                        else:
                            dd_stats.increment(dd_prefix + "dataprep.gave_up")
                            logging.error(
                                "Error %r from dataprep server for paper id %s. Giving up.",
                                e,
                                paper_id)
                            error = {
                                "error": {
                                    "message": "Error %r while contacting dataprep server" % e,
                                    "stackTrace": None,
                                    "docName": "%s.pdf" % paper_id
                                }
                            }
                            write_json_to_output(error)
                            break

                # append the tempfile to the json file
                f.flush()
                f.seek(0)
                async with write_lock:
                    _send_all(f, json_file)

        while True:
            paper_sha_to_messages = get_pdf_hash_to_messages(queue_client, queue_url, corpus_db, dd_stats, dd_prefix)
            paper_shas = list(paper_sha_to_messages.keys())

            temp_dir = tempfile.TemporaryDirectory(prefix="SPv2DBWorker-")

            logging.info("Getting JSON ...")
            getting_json_time = time.time()
            json_file_name = os.path.join(temp_dir.name, "tokens.json")
            with open(json_file_name, "wb") as json_file:
                write_json_futures = [write_json_tokens_to_file(p, json_file) for p in paper_shas]
                async_event_loop.run_until_complete(asyncio.wait(write_json_futures))
            getting_json_time = time.time() - getting_json_time
            logging.info("Got JSON in %.2f seconds", getting_json_time)
            dd_stats.timing(dd_prefix + "get_json", getting_json_time)

            # pick out errors and write them to the DB
            paper_id_to_error = {}
            for line in dataprep2.json_from_file(json_file_name):
                if not "error" in line:
                    continue
                error = line["error"]
                error["message"] = dataprep2.sanitize_for_json(error["message"])
                error["stackTrace"] = dataprep2.sanitize_for_json(error["stackTrace"])
                paper_id = error["docName"]
                if paper_id.endswith(".pdf"):
                    paper_id = paper_id[:-4]
                paper_id_to_error[paper_id] = error
                logging.error("Paper %s has error: %s", paper_id, error["message"])

            dd_stats.increment(dd_prefix + "errors", len(paper_id_to_error))

            # # make unlabeled tokens file
            # logging.info("Making unlabeled tokens ...")
            # making_unlabeled_tokens_time = time.time()
            # unlabeled_tokens_file_name = os.path.join(temp_dir.name, "unlabeled-tokens.h5")
            # dataprep2.make_unlabeled_tokens_file(
            #     json_file_name,
            #     unlabeled_tokens_file_name,
            #     ignore_errors=True)
            # os.remove(json_file_name)
            # making_unlabeled_tokens_time = time.time() - making_unlabeled_tokens_time
            # logging.info("Made unlabeled tokens in %.2f seconds", making_unlabeled_tokens_time)
            # dd_stats.timing(dd_prefix + "make_unlabeled", making_unlabeled_tokens_time)
            #
            # # make featurized tokens file
            # logging.info("Making featurized tokens ...")
            # making_featurized_tokens_time = time.time()
            # with h5py.File(unlabeled_tokens_file_name, "r") as unlabeled_tokens_file:
            #     featurized_tokens_file_name = os.path.join(temp_dir.name, "featurized-tokens.h5")
            #     dataprep2.make_featurized_tokens_file(
            #         featurized_tokens_file_name,
            #         unlabeled_tokens_file,
            #         token_stats,
            #         embeddings,
            #         dataprep2.VisionOutput(None),
            #         model_settings
            #     )
            #     # We don't delete the unlabeled file here because the featurized one contains references
            #     # to it.
            # making_featurized_tokens_time = time.time() - making_featurized_tokens_time
            # logging.info("Made featurized tokens in %.2f seconds", making_featurized_tokens_time)
            # dd_stats.timing(dd_prefix + "make_featurized", making_featurized_tokens_time)

            yield temp_dir, featurized_tokens_file_name, paper_sha_to_messages

    logging.info("Starting to process tasks")
    for temp_dir, featurized_tokens_file_name, paper_sha_to_messages in dataprep2.threaded_generator(featurized_tokens_filenames(), 1):
        try:
            logging.info("Making and sending results ...")
            make_and_send_results_time = time.time()
            with h5py.File(featurized_tokens_file_name) as featurized_tokens_file:
                # def get_docs():
                #     return dataprep2.documents_for_featurized_tokens(
                #         featurized_tokens_file,
                #         include_labels=False,
                #         max_tokens_per_page=model_settings.tokens_per_batch)
                #
                # model_results = with_labels.run_model(
                #     model,
                #     model_settings,
                #     embeddings.glove_vocab(),
                #     get_docs,
                #     enabled_modes={"predictions"})
                #
                # results = {
                #     doc.doc_sha: {
                #         "docName": doc.doc_id,
                #         "docSha": doc.doc_sha,
                #         "title": dataprep2.sanitize_for_json(docresults["predictions"][0]),
                #         "authors": docresults["predictions"][1],
                #         "bibs": [
                #             {
                #                 "title": bibtitle,
                #                 "authors": bibauthors,
                #                 "venue": bibvenue,
                #                 "year": bibyear
                #             } for bibtitle, bibauthors, bibvenue, bibyear in docresults["predictions"][2]
                #         ]
                #     } for doc, docresults in model_results
                # }
                results = [{'foo': 'bar'}]
                dd_stats.increment(dd_prefix + "successes", len(results))
                # TODO send (pdf hash, model version, sp version, result) to ParseStore

                # Delete successfully processed messages from queue
                messages_to_delete = [paper_sha_to_messages[sha] for sha in list(results.keys())]
                logging.info('Deleting messages')
                delete_batch(queue_client, queue_url, messages_to_delete)
        finally:
            temp_dir.cleanup()

        make_and_send_results_time = time.time() - make_and_send_results_time
        logging.info("Made and sent results in %.2f seconds", make_and_send_results_time)
        dd_stats.timing(dd_prefix + "make_results", make_and_send_results_time)

if __name__ == "__main__":
    main()
