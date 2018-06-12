import logging
import os
import time
import typing
import asyncio
import aiohttp
import json
import papertasks

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

def main():
    import tempfile
    import argparse
    import h5py
    import datadog

    import settings
    import dataprep2

    if os.name != 'nt':
        import manhole
        manhole.install()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(thread)d %(levelname)s %(message)s', level=logging.INFO)

    default_password = os.environ.get("CORPUSDB_PASSWORD")
    default_dataprep_host = os.environ.get("SPV2_DATAPREP_SERVICE_HOST", "localhost")
    default_dataprep_port = int(os.environ.get("SPV2_DATAPREP_SERVICE_PORT", "8080"))
    parser = argparse.ArgumentParser(description="Trains a classifier for PDF Tokens")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="database host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5432,
        help="database port"
    )
    parser.add_argument(
        "--dbname",
        type=str,
        default="corpus",
        help="database name"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="public",
        help="schema name"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="s2dev",
        help="database user"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=default_password,
        help="database password"
    )
    parser.add_argument(
        "--dataprep-host",
        type=str,
        default=default_dataprep_host,
        help="Host where the dataprep service is running"
    )
    parser.add_argument(
        "--dataprep-port",
        type=str,
        default=default_dataprep_port,
        help="Port where the dataprep service is running"
    )
    args = parser.parse_args()

    todo_list = papertasks.TaskDB(
        host = args.host,
        port = args.port,
        dbname = args.dbname,
        schema = args.schema,
        user = args.user,
        password = args.password
    )

    # start datadog
    datadog.initialize(api_key=os.environ.get("DATADOG_API_KEY"))
    stats = datadog.ThreadStats()
    stats.start()
    datadog_prefix = args.host.split(".")[0]
    if datadog_prefix.startswith("spv2-"):
        datadog_prefix = datadog_prefix[5:]
    datadog_prefix = "spv2.%s." % datadog_prefix

    logging.info("Loading model settings ...")
    model_settings = settings.default_model_settings

    logging.info("Loading token statistics ...")
    token_stats = dataprep2.TokenStatistics("model/all.tokenstats3.gz")

    logging.info("Loading embeddings ...")
    embeddings = dataprep2.CombinedEmbeddings(
        token_stats,
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.embedded_tokens_fraction
    )

    import with_labels  # Heavy import, so we do it here
    model = with_labels.model_with_labels(model_settings, embeddings)
    model.load_weights("model/C49.h5")
    model_version = 2

    logging.info("Starting to process tasks")
    total_paper_ids_processed = 0
    start_time = time.time()
    last_time_with_paper_ids = start_time

    def featurized_tokens_filenames() -> typing.Generator[typing.Tuple[tempfile.TemporaryDirectory, str], None, None]:
        # async http stuff
        async_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_event_loop)
        connector = aiohttp.TCPConnector(loop=async_event_loop, force_close=True)
        session = aiohttp.ClientSession(connector=connector, read_timeout=120, conn_timeout=120)
        write_lock = asyncio.Lock()
        async def write_json_tokens_to_file(paper_id: str, json_file):
            url = "http://%s:%d/v1/json/paperid/%s" % (args.dataprep_host, args.dataprep_port, paper_id)
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
                                stats.increment(datadog_prefix + "dataprep.success")
                                break
                            else:
                                stats.increment(datadog_prefix + "dataprep.failure")
                                if attempts_left > 0:
                                    logging.error(
                                        "Error %d from dataprep server for paper id %s. %d attempts left.",
                                        response.status,
                                        paper_id,
                                        attempts_left)
                                else:
                                    stats.increment(datadog_prefix + "dataprep.gave_up")
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
                        stats.increment(datadog_prefix + "dataprep.failure")
                        if attempts_left > 0:
                            logging.error(
                                "Error %r from dataprep server for paper id %s. %d attempts left.",
                                e,
                                paper_id,
                                attempts_left)
                        else:
                            stats.increment(datadog_prefix + "dataprep.gave_up")
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
                with await write_lock:
                    _send_all(f, json_file)

        processing_timeout = 600
        while True:
            paper_ids = todo_list.get_batch_to_process(model_version, max_batch_size=50)
            logging.info("Received %d paper ids", len(paper_ids))
            if len(paper_ids) <= 0:
                if time.time() - last_time_with_paper_ids > processing_timeout:
                    logging.info("Saw no paper ids for more than %.0f seconds. Shutting down.", processing_timeout)
                    return
                time.sleep(20)
                continue
            stats.increment(datadog_prefix + "attempts", len(paper_ids))

            temp_dir = tempfile.TemporaryDirectory(prefix="SPv2DBWorker-")

            logging.info("Getting JSON ...")
            getting_json_time = time.time()
            json_file_name = os.path.join(temp_dir.name, "tokens.json")
            with open(json_file_name, "wb") as json_file:
                write_json_futures = [write_json_tokens_to_file(p, json_file) for p in paper_ids]
                async_event_loop.run_until_complete(asyncio.wait(write_json_futures))
            getting_json_time = time.time() - getting_json_time
            logging.info("Got JSON in %.2f seconds", getting_json_time)
            stats.timing(datadog_prefix + "get_json", getting_json_time)

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
                logging.info("Paper %s has error %s", paper_id, error["message"])
            if len(paper_id_to_error) > len(paper_ids) / 2:
                raise ValueError("More than half of the batch failed to preprocess. Something is afoot. We're giving up.")
            todo_list.post_errors(model_version, paper_id_to_error)
            stats.increment(datadog_prefix + "errors", len(paper_id_to_error))
            logging.info("Wrote %d errors to database", len(paper_id_to_error))

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
            stats.timing(datadog_prefix + "make_unlabeled", making_unlabeled_tokens_time)

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
                # We don't delete the unlabeled file here because the featurized one contains references
                # to it.
            making_featurized_tokens_time = time.time() - making_featurized_tokens_time
            logging.info("Made featurized tokens in %.2f seconds", making_featurized_tokens_time)
            stats.timing(datadog_prefix + "make_featurized", making_featurized_tokens_time)

            yield temp_dir, featurized_tokens_file_name

    for temp_dir, featurized_tokens_file_name in dataprep2.threaded_generator(featurized_tokens_filenames(), 1):
        try:
            logging.info("Making and sending results ...")
            make_and_send_results_time = time.time()
            with h5py.File(featurized_tokens_file_name) as featurized_tokens_file:
                def get_docs():
                    return dataprep2.documents_for_featurized_tokens(
                        featurized_tokens_file,
                        include_labels=False,
                        max_tokens_per_page=model_settings.tokens_per_batch)
                results = with_labels.run_model(
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
                    } for doc, docresults in results
                }

                todo_list.post_results(model_version, results)
                stats.increment(datadog_prefix + "successes", len(results))
                total_paper_ids_processed += len(results)
        finally:
            temp_dir.cleanup()

        make_and_send_results_time = time.time() - make_and_send_results_time
        logging.info("Made and sent results in %.2f seconds", make_and_send_results_time)
        stats.timing(datadog_prefix + "make_results", make_and_send_results_time)

        # report progress
        paper_ids_per_hour = 3600 * total_paper_ids_processed / (time.time() - start_time)
        logging.info("This worker is processing %.0f paper ids per hour." % paper_ids_per_hour)

        last_time_with_paper_ids = time.time()

if __name__ == "__main__":
    main()
