import logging
import psycopg2
import psycopg2.extras
import os
import time

class DBTodoList:
    EXPECTED_VERSION = 1

    def __init__(self, **kwargs):
        self.conn = psycopg2.connect(**kwargs)

        # find out the version we're at
        with self.conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS settings (key VARCHAR PRIMARY KEY, value VARCHAR);")

            cur.execute("SELECT value FROM settings WHERE key = 'version';")
            version = cur.fetchone()[0]

            if version is not None:
                version = int(version)
            else:
                logging.info("Updating database to version 1")

                try:
                    cur.execute("""
                        CREATE TYPE processing_status AS ENUM (
                          'Scheduled',
                          'Processing',
                          'Done',
                          'Failed'
                        );
                    """)

                    cur.execute("""
                        CREATE TABLE tasks (
                          modelVersion SMALLINT NOT NULL,
                          paperId CHAR(40) NOT NULL,
                          status processing_status NOT NULL DEFAULT 'Scheduled'::processing_status,
                          statusChanged TIMESTAMP NOT NULL DEFAULT NOW(),
                          attempts SMALLINT NOT NULL DEFAULT 0,
                          result JSONB,
                          PRIMARY KEY (modelVersion, paperId)
                        );
                    """)

                    cur.execute("""
                        CREATE FUNCTION update_status_changed_trigger() RETURNS TRIGGER AS $$
                          BEGIN
                              NEW.statusChanged := NOW();
                              RETURN NEW;
                          END
                        $$ LANGUAGE plpgsql;
                    """)

                    cur.execute("""
                        CREATE TRIGGER update_status_changed
                        BEFORE UPDATE OF status ON tasks
                        FOR EACH ROW
                        EXECUTE PROCEDURE update_status_changed_trigger();
                    """)

                    cur.execute("""
                        CREATE FUNCTION effective_status_fn (
                          status processing_status,
                          statusChanged TIMESTAMP,
                          attempts SMALLINT
                        ) RETURNS processing_status
                        RETURNS NULL ON NULL INPUT
                        AS
                        $$
                        SELECT CASE
                          WHEN
                            -- scheduled more than five times
                            status = 'Scheduled'::processing_status AND
                            attempts >= 5
                            THEN 'Failed'::processing_status
                          WHEN
                            -- processing expired, and scheduled more than five times
                            status = 'Processing'::processing_status AND
                            statusChanged + interval '5 minutes' < NOW() AND
                            attempts >= 5
                            THEN 'Failed'::processing_status
                          WHEN
                            -- processing expired
                            status = 'Processing'::processing_status AND
                            statusChanged + interval '5 minutes' < NOW()
                            THEN 'Scheduled'::processing_status                        
                          ELSE status 
                        END
                        $$ LANGUAGE SQL IMMUTABLE;
                    """)

                    cur.execute("""
                        CREATE VIEW tasks_with_status AS
                          SELECT *, effective_status_fn(status, statusChanged, attempts) 
                          AS effectiveStatus
                          FROM tasks;
                    """)

                    cur.execute("""
                        CREATE INDEX ON tasks(modelversion, status, paperid);
                    """)

                    # set the version number
                    cur.execute("INSERT INTO settings (key, value) VALUES ('version', 1);")
                    self.conn.commit()
                    version = 1
                except:
                    self.conn.rollback()
                    raise
            logging.info("Database is at version %d", version)

    def get_batch_to_process(self, model_version: int, max_batch_size: int=100):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    WITH selected AS (
                      SELECT modelversion, paperid
                      FROM tasks_with_status
                      WHERE
                        ( -- not necessary, but should result in better use of the index
                          status = 'Scheduled'::PROCESSING_STATUS OR
                          status = 'Processing'::PROCESSING_STATUS
                        ) AND
                        effectivestatus = 'Scheduled'::processing_status AND
                        modelversion = %s
                      LIMIT %s
                      FOR UPDATE SKIP LOCKED
                    )
                    UPDATE tasks_with_status ts
                    SET
                      status = 'Processing'::processing_status,
                      attempts = ts.attempts + 1
                    FROM selected
                    WHERE
                      selected.paperid = ts.paperid AND
                      selected.modelversion = ts.modelversion
                    RETURNING ts.paperid
                """, (model_version, max_batch_size))

                paper_ids = cur.fetchall()
            self.conn.commit()
        except:
            self.conn.rollback()
            raise

        return [x[0] for x in paper_ids]

    def post_results(self, model_version: int, paper_id_to_result):
        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """INSERT INTO tasks (modelversion, paperid, status, attempts, result)
                    VALUES (%s, %s, 'Done'::processing_status, 1, %s)
                    ON CONFLICT (modelversion, paperid) DO UPDATE SET 
                      status = EXCLUDED.status,
                      result = EXCLUDED.result""",
                    [
                        (model_version, pid, psycopg2.extras.Json(result))
                        for pid, result in paper_id_to_result.items()
                    ]
                )
            self.conn.commit()
        except:
            self.conn.rollback()
            raise

    def post_errors(self, model_version: int, paper_id_to_error):
        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """INSERT INTO tasks (modelversion, paperid, status, attempts, result)
                    VALUES (%s, %s, 'Scheduled'::processing_status, 1, %s)
                    ON CONFLICT (modelversion, paperid) DO UPDATE SET 
                      status = EXCLUDED.status,
                      result = EXCLUDED.result""",
                    [
                        (model_version, pid, psycopg2.extras.Json(result))
                        for pid, result in paper_id_to_error.items()
                    ]
                )
            self.conn.commit()
        except:
            self.conn.rollback()
            raise

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
    import http.client
    import h5py

    import settings
    import dataprep2

    if os.name != 'nt':
        import manhole
        manhole.install()

    logging.getLogger().setLevel(logging.DEBUG)

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
        default="spv2",
        help="database name"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="database user"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="database password"
    )
    args = parser.parse_args()

    todo_list = DBTodoList(**args.__dict__)

    logging.info("Loading model settings ...")
    model_settings = settings.default_model_settings

    logging.info("Loading token statistics ...")
    token_stats = dataprep2.TokenStatistics("model/all.tokenstats3.gz")

    logging.info("Loading embeddings ...")
    embeddings = dataprep2.CombinedEmbeddings(
        token_stats,
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.minimum_token_frequency
    )

    import with_labels  # Heavy import, so we do it here
    model = with_labels.model_with_labels(model_settings, embeddings)
    model.load_weights("model/B40.h5")
    model_version = 1

    logging.info("Starting to process tasks")
    total_paper_ids_processed = 0
    start_time = time.time()
    last_time_with_paper_ids = start_time
    processing_timeout = 600
    while True:
        paper_ids = todo_list.get_batch_to_process(model_version)
        logging.info("Received %d paper ids", len(paper_ids))
        if len(paper_ids) <= 0:
            if time.time() - last_time_with_paper_ids > processing_timeout:
                logging.info("Saw no paper ids for more than %.0f seconds. Shutting down.", processing_timeout)
                return
            time.sleep(20)
            continue

        # make URLs from the paperids
        json_request_body = []
        s3_url_to_paper_id = {}
        templates = ["s3://ai2-s2-pdfs/%s/%s.pdf", "s3://ai2-s2-pdfs-private/%s/%s.pdf"]
        for paper_id in paper_ids:
            json_request_line = []
            for template in templates:
                url = template % (paper_id[:4], paper_id[4:])
                json_request_line.append(url)
                s3_url_to_paper_id[url] = paper_id
            json_request_line = " ".join(json_request_line)
            json_request_body.append(json_request_line)
        json_request_body = "\n".join(json_request_body)

        with tempfile.TemporaryDirectory(prefix="SPV2DBWorker-") as temp_dir:
            # make JSON out of the papers
            logging.info("Getting JSON ...")
            getting_json_time = time.time()
            json_file_name = os.path.join(temp_dir, "tokens.json")
            with open(json_file_name, "wb") as json_file:
                dataprep_conn = http.client.HTTPConnection("localhost", 8080, timeout=600)
                dataprep_conn.request("POST", "/v1/json/urls", body=json_request_body)
                with dataprep_conn.getresponse() as dataprep_response:
                    if dataprep_response.status < 200 or dataprep_response.status >= 300:
                        raise ValueError("Error %d from dataprep server at %s" % (
                            dataprep_response.status,
                            dataprep_conn.host))
                    _send_all(dataprep_response, json_file)
            getting_json_time = time.time() - getting_json_time
            logging.info("Got JSON in %.2f seconds", getting_json_time)

            # pick out errors and write them to the DB
            paper_id_to_error = {}
            for line in dataprep2.json_from_file(json_file_name):
                if not "error" in line:
                    continue
                line = line["error"]
                paper_id = line["docName"]
                paper_id = s3_url_to_paper_id[paper_id]
                paper_id_to_error[paper_id] = line
            todo_list.post_errors(model_version, paper_id_to_error)

            # make unlabeled tokens file
            logging.info("Making unlabeled tokens ...")
            making_unlabeled_tokens_time = time.time()
            unlabeled_tokens_file_name = os.path.join(temp_dir, "unlabeled-tokens.h5")
            dataprep2.make_unlabeled_tokens_file(
                json_file_name,
                unlabeled_tokens_file_name,
                ignore_errors=True)
            os.remove(json_file_name)
            making_unlabeled_tokens_time = time.time() - making_unlabeled_tokens_time
            logging.info("Made unlabeled tokens in %.2f seconds", making_unlabeled_tokens_time)

            # make featurized tokens file
            logging.info("Making featurized tokens ...")
            making_featurized_tokens_time = time.time()
            with h5py.File(unlabeled_tokens_file_name, "r") as unlabeled_tokens_file:
                featurized_tokens_file_name = os.path.join(temp_dir, "featurized-tokens.h5")
                dataprep2.make_featurized_tokens_file(
                    featurized_tokens_file_name,
                    unlabeled_tokens_file,
                    token_stats,
                    embeddings,
                    dataprep2.VisionOutput(None),   # TODO: put in real vision output
                    model_settings
                )
                # We don't delete the unlabeled file here because the featurized one contains references
                # to it.
            making_featurized_tokens_time = time.time() - making_featurized_tokens_time
            logging.info("Made featurized tokens in %.2f seconds", making_featurized_tokens_time)

            logging.info("Making and sending results ...")
            make_and_send_results_time = time.time()
            with h5py.File(featurized_tokens_file_name) as featurized_tokens_file:
                def get_docs():
                    return dataprep2.documents_for_featurized_tokens(
                        featurized_tokens_file,
                        include_labels=False,
                        max_tokens_per_page=model_settings.tokens_per_batch)
                results = with_labels.run_model(model, model_settings, get_docs)
                results = {
                    doc.doc_sha: {
                        "docName": doc.doc_id,
                        "docSha": doc.doc_sha,
                        "title": title,
                        "authors": authors
                    } for doc, title, authors in results
                }

                todo_list.post_results(model_version, results)
                total_paper_ids_processed += len(results)

            make_and_send_results_time = time.time() - make_and_send_results_time
            logging.info("Made and sent results in %.2f seconds", make_and_send_results_time)

        # report progress
        paper_ids_per_hour = 3600 * total_paper_ids_processed / (time.time() - start_time)
        logging.info("This worker is processing %.0f paper ids per hour." % paper_ids_per_hour)

        last_time_with_paper_ids = time.time()

if __name__ == "__main__":
    main()
