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
import with_labels


class QueueWorker:
    """
    1. Pulls modified SourcedPaper events from queue
    2. Gets PDF of SourcedPaper
    3. Runs model on PDF
    4. Puts result in ParseStore
    5. Publishes event to downstream SNS topic
    """
    SLEEP_SECONDS = 10
    MODEL_VERSION = 2
    MAX_NUMBER_OF_MESSAGES = 10  # upper limit for SQS
    VISIBILITY_TIMEOUT = 600  # time till messages become visible in queue again, in seconds
    POLL_TIMEOUT = 20  # timeout for SQS long polling, in seconds

    def __init__(self, env, queue_name, dataprep_host, dataprep_port, corpus_db, dd_api_key, sns_topic):
        self.queue_client = boto3.client("sqs", region_name="us-west-2")
        self.queue_url = self.queue_client.get_queue_url(QueueName=queue_name)["QueueUrl"]

        self.dataprep_host = dataprep_host
        self.dataprep_port = dataprep_port

        self.corpus_db = corpus_db

        datadog.initialize(api_key=dd_api_key)
        self.dd_stats = datadog.ThreadStats()
        self.dd_stats.start()
        self.dd_prefix = "spv2.%s." % env

        self.sns_topic = sns_topic

    def process_queue(self, model_settings, token_stats, embeddings, model):
        feat_token_filenames = dataprep2.threaded_generator(self._generate_featurized_tokens_filenames(token_stats, embeddings, model_settings), 1)
        for temp_dir, featurized_tokens_filename, pdf_hash_to_messages in feat_token_filenames:
            try:
                logging.info("Making and sending results ...")
                make_and_send_results_time = time.time()
                parse_results = self.parse_featurized_tokens_file(featurized_tokens_filename, model_settings, model, embeddings)
                self.dd_stats.increment(self.dd_prefix + "successes", len(parse_results))
                # TODO send (pdf hash, model version, sp version, result) to ParseStore

                # Delete successfully processed messages from queue
                logging.info('results: %s', parse_results)
                messages_to_delete = [m for sha in parse_results.keys() for m in pdf_hash_to_messages[sha]]
                self.delete_batch(self.queue_client, self.queue_url, messages_to_delete)
            finally:
                temp_dir.cleanup()

            make_and_send_results_time = time.time() - make_and_send_results_time
            logging.info("Made and sent results in %.2f seconds", make_and_send_results_time)
            self.dd_stats.timing(self.dd_prefix + "make_results", make_and_send_results_time)

    @staticmethod
    def parse_featurized_tokens_file(filename, model_settings, model, embeddings):
        with h5py.File(filename) as featurized_tokens_file:
            def get_docs():
                return dataprep2.documents_for_featurized_tokens(
                    featurized_tokens_file,
                    include_labels=False,
                    max_tokens_per_page=model_settings.tokens_per_batch)

            model_results = with_labels.run_model(
                model,
                model_settings,
                embeddings.glove_vocab(),
                get_docs,
                enabled_modes={"predictions"})

            results = {
                doc.doc_sha: {
                    "docName": doc.doc_id,
                    "docSha": doc.doc_sha,
                    "title": dataprep2.sanitize_for_json(docresults["predictions"][0]),
                    "authors": docresults["predictions"][1],
                    "bibs": [
                        {
                            "title": bibtitle,
                            "authors": bibauthors,
                            "venue": bibvenue,
                            "year": bibyear
                        } for bibtitle, bibauthors, bibvenue, bibyear in docresults["predictions"][2]
                    ]
                } for doc, docresults in model_results
            }
            return results

    def _generate_featurized_tokens_filenames(self, token_stats, embeddings, model_settings):
        # async http stuff
        async_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_event_loop)
        connector = aiohttp.TCPConnector(loop=async_event_loop, force_close=True)
        session = aiohttp.ClientSession(connector=connector, read_timeout=120, conn_timeout=120)
        write_lock = asyncio.Lock()

        while True:
            pdf_hash_to_messages = self._get_pdf_hash_to_queue_messages()
            pdf_hashes = list(pdf_hash_to_messages.keys())

            temp_dir = tempfile.TemporaryDirectory(prefix="SPv2QueueWorker-")

            logging.info("Getting JSON ...")
            getting_json_time = time.time()
            json_file_name = os.path.join(temp_dir.name, "tokens.json")
            with open(json_file_name, "wb") as json_file:
                write_json_futures = [self._write_json_tokens_to_file(p, json_file, session, write_lock) for p in pdf_hashes]
                async_event_loop.run_until_complete(asyncio.wait(write_json_futures))
            getting_json_time = time.time() - getting_json_time
            logging.info("Got JSON in %.2f seconds", getting_json_time)
            self.dd_stats.timing(self.dd_prefix + "get_json", getting_json_time)

            # pick out errors and write them to the DB
            paper_id_to_error = {}
            for line in dataprep2.json_from_file(json_file_name):
                if "error" not in line:
                    continue
                error = line["error"]
                error["message"] = dataprep2.sanitize_for_json(error["message"])
                error["stackTrace"] = dataprep2.sanitize_for_json(error["stackTrace"])
                paper_id = error["docName"]
                if paper_id.endswith(".pdf"):
                    paper_id = paper_id[:-4]
                paper_id_to_error[paper_id] = error
                logging.error("Paper %s has error: %s", paper_id, error["message"])

            self.dd_stats.increment(self.dd_prefix + "errors", len(paper_id_to_error))

            # make unlabeled tokens file
            logging.info("Making unlabeled tokens ...")
            making_unlabeled_tokens_time = time.time()
            unlabeled_tokens_file_name = os.path.join(temp_dir.name, "unlabeled-tokens.h5")
            dataprep2.make_unlabeled_tokens_file(
                json_file_name,
                unlabeled_tokens_file_name,
                ignore_errors=True)
            os.remove(json_file_name)
            making_unlabeled_tokens_time = time.time() - making_unlabeled_tokens_time
            logging.info("Made unlabeled tokens in %.2f seconds", making_unlabeled_tokens_time)
            self.dd_stats.timing(self.dd_prefix + "make_unlabeled", making_unlabeled_tokens_time)

            # make featurized tokens file
            logging.info("Making featurized tokens ...")
            making_featurized_tokens_time = time.time()
            with h5py.File(unlabeled_tokens_file_name, "r") as unlabeled_tokens_file:
                featurized_tokens_file_name = os.path.join(temp_dir.name, "featurized-tokens.h5")
                dataprep2.make_featurized_tokens_file(
                    featurized_tokens_file_name,
                    unlabeled_tokens_file,
                    token_stats,
                    embeddings,
                    dataprep2.VisionOutput(None),
                    model_settings
                )
            # We don't delete the unlabeled file here because the featurized one contains references to it.
            making_featurized_tokens_time = time.time() - making_featurized_tokens_time
            logging.info("Made featurized tokens in %.2f seconds", making_featurized_tokens_time)
            self.dd_stats.timing(self.dd_prefix + "make_featurized", making_featurized_tokens_time)

            yield temp_dir, featurized_tokens_file_name, pdf_hash_to_messages

    def _get_pdf_hash_to_queue_messages(self):
        while True:
            messages = self.queue_client.receive_message(
                QueueUrl=self.queue_url,
                VisibilityTimeout=self.VISIBILITY_TIMEOUT,
                MaxNumberOfMessages=self.MAX_NUMBER_OF_MESSAGES,
                WaitTimeSeconds=self.POLL_TIMEOUT
            ).get('Messages', [])

            if len(messages) == 0:
                logging.info("No papers to process. Sleeping for %s", str(self.SLEEP_SECONDS))
                time.sleep(self.SLEEP_SECONDS)
                continue

            messages_to_delete = []
            # Need this mapping because of downstream batch processing and we lose paper id to message id
            # correspondence. We need message ids to delete success events from queue.
            pdf_hash_to_messages = dict()
            for m in messages:
                event = self.parse_event(m)
                if not event or (event.type, event.action) != ('SourcedPaper', 'modified'):
                    messages_to_delete.append(m)
                else:
                    sha = self.corpus_db.get_sourced_paper_pdf_hash(event.id)
                    if sha:
                        # TODO check ParseStore and skip if (PDF hash, model version, SP version) already exists
                        if sha in pdf_hash_to_messages:
                            pdf_hash_to_messages[sha].append(m)
                        else:
                            pdf_hash_to_messages[sha] = [m]
                    else:
                        messages_to_delete.append(m)

            self.delete_batch(self.queue_client, self.queue_url, messages_to_delete)
            num_to_process = len(pdf_hash_to_messages)

            if num_to_process > 0:
                self.dd_stats.increment(self.dd_prefix + "attempts", num_to_process)
                return pdf_hash_to_messages

    async def _write_json_tokens_to_file(self, paper_id: str, json_file, session, write_lock):
        url = "http://%s:%d/v1/json/paperid/%s" % (self.dataprep_host, self.dataprep_port, paper_id)
        attempts_left = 5
        with tempfile.NamedTemporaryFile(prefix="SPv2QueueWorker-%s-" % paper_id, suffix=".json") as f:
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
                            self.dd_stats.increment(self.dd_prefix + "dataprep.success")
                            break
                        else:
                            self.dd_stats.increment(self.dd_prefix + "dataprep.failure")
                            if attempts_left > 0:
                                logging.error(
                                    "Error %d from dataprep server for paper id %s. %d attempts left.",
                                    response.status,
                                    paper_id,
                                    attempts_left)
                            else:
                                self.dd_stats.increment(self.dd_prefix + "dataprep.gave_up")
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
                    self.dd_stats.increment(self.dd_prefix + "dataprep.failure")
                    if attempts_left > 0:
                        logging.error(
                            "Error %r from dataprep server for paper id %s. %d attempts left.",
                            e,
                            paper_id,
                            attempts_left)
                    else:
                        self.dd_stats.increment(self.dd_prefix + "dataprep.gave_up")
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
                self.send_all(f, json_file)

    @staticmethod
    def send_all(source, dest, nbytes: int = None):
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

    @staticmethod
    def parse_event(message):
        try:
            event = Event(**json.loads(message['Body']))
            return event
        except Exception:
            logging.exception('Failed parsing message from queue. Message: %s', str(message))
            return None

    @staticmethod
    def delete_batch(queue_client, queue_url, messages_to_delete):
        if messages_to_delete:
            queue_client.delete_message_batch(
                QueueUrl=queue_url,
                Entries=[{
                    'Id': message['MessageId'],
                    'ReceiptHandle': message['ReceiptHandle']
                } for message in messages_to_delete]
            )
