import logging
import os

import dataprep2
import settings
from queue_processor.corpus_db import CorpusDB
from queue_processor.queue_worker import QueueWorker


if __name__ == "__main__":
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
    logging.info("Env: %s", env)

    upstream_queue_name = os.environ["UPSTREAM_QUEUE_NAME"]
    logging.info("Upstream SQS queue name: %s", upstream_queue_name)

    dd_api_key = os.environ["DATADOG_API_KEY"]
    downstream_queue_name = os.environ["DOWNSTREAM_QUEUE_NAME"]
    logging.info("Downstream SQS queue name: %s", downstream_queue_name)

    queue_worker = QueueWorker(
        env=env,
        upstream_queue_name=upstream_queue_name,
        dataprep_host=dataprep_host,
        dataprep_port=dataprep_port,
        corpus_db=corpus_db,
        dd_api_key=dd_api_key,
        downstream_queue_name=downstream_queue_name
    )
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

    logging.info("Processing queue")
    queue_worker.process_queue(model_settings, token_stats, embeddings, model)
