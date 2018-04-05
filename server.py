#!/usr/bin/env python
# -*- coding: utf8 -*-

import http.server
import http.client
import tempfile
import os
import h5py
import logging
import json
import codecs
import time
import re

import dataprep2
import settings
import with_labels


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

class RequestHandler(http.server.BaseHTTPRequestHandler):
    _url_re = re.compile("/v1/json/paperid/([a-z0-9]{40})")

    def do_GET(self):
        m = self._url_re.match(self.path)
        if m is None:
            self.send_error(404)
            return

        paper_id = m.groups(1)
        with tempfile.TemporaryDirectory(prefix="SPV2Server-") as temp_dir:
            logging.info("Getting JSON ...")
            getting_json_time = time.time()
            # get json from the dataprep server
            json_file_name = os.path.join(temp_dir, "tokens.json")
            with open(json_file_name, "wb") as json_file:
                dataprep_conn = http.client.HTTPConnection("localhost", 8080, timeout=60)
                dataprep_conn.request("GET", "/v1/json/paperid/%s" % paper_id)
                with dataprep_conn.getresponse() as dataprep_response:
                    if dataprep_response.status < 200 or dataprep_response.status >= 300:
                        raise ValueError("Error %d from dataprep server at %s" % (
                            dataprep_response.status,
                            dataprep_conn.host))
                    _send_all(dataprep_response, json_file)
            getting_json_time = time.time() - getting_json_time
            logging.info("Got JSON in %.2f seconds", getting_json_time)

            # make unlabeled tokens file
            logging.info("Making unlabeled tokens ...")
            making_unlabeled_tokens_time = time.time()
            unlabeled_tokens_file_name = os.path.join(temp_dir, "unlabeled-tokens.h5")
            dataprep2.make_unlabeled_tokens_file(
                json_file_name,
                unlabeled_tokens_file_name,
                ignore_errors=True)
            errors = [line for line in dataprep2.json_from_file(json_file_name) if "error" in line]
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
                    self.server.token_stats,
                    self.server.embeddings,
                    dataprep2.VisionOutput(None),
                    self.server.model_settings
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
                        max_tokens_per_page=self.server.model_settings.tokens_per_batch)
                results = with_labels.run_model(
                    self.server.model,
                    self.server.model_settings,
                    self.server.embeddings.glove_vocab(),
                    get_docs,
                    enabled_modes={"predictions"})

                started_sending = False
                response_body = codecs.getwriter("UTF-8")(self.wfile, "UTF-8")
                for doc, docresults in results:
                    result_json = {
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
                    }
                    result_json = {"doc": result_json}

                    if not started_sending:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        started_sending = True

                    json.dump(result_json, response_body)
                    response_body.write("\n")
                for error in errors:
                    json.dump(error, response_body)
                    response_body.write("\n")

                response_body.reset()
            make_and_send_results_time = time.time() - make_and_send_results_time
            logging.info("Made and sent results in %.2f seconds", make_and_send_results_time)

            logging.info("Done processing")
            logging.info("Getting JSON:          %.0f s", getting_json_time)
            logging.info("Unlabeled tokens:      %.0f s", making_unlabeled_tokens_time)
            logging.info("Featurized tokens:     %.0f s", making_featurized_tokens_time)
            logging.info("Make and send results: %.0f s", make_and_send_results_time)


    def do_PUT(self):
        self.send_error(405)

    def do_DELETE(self):
        self.send_error(405)

    def do_PATCH(self):
        self.send_error(405)

    def do_POST(self):
        self.send_error(405)

class Server(http.server.HTTPServer):
    def __init__(self, model, token_stats: dataprep2.TokenStatistics, embeddings: dataprep2.CombinedEmbeddings, model_settings):
        super(Server, self).__init__(('', 8081), RequestHandler)

        self.model = model
        self.model._make_predict_function()
        self.token_stats = token_stats
        self.embeddings = embeddings
        self.model_settings = model_settings

        self.token_stats._ensure_loaded()
        self.embeddings._ensure_loaded()


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    if os.name != 'nt':
        import manhole
        manhole.install()

    model_settings = settings.default_model_settings

    import argparse
    parser = argparse.ArgumentParser(description="Runs the SPv2 server")
    parser.add_argument(
        "--tokens-per-batch",
        type=int,
        default=model_settings.tokens_per_batch,
        help="the number of tokens in a batch"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/C42.h5",
        help="filename of existing model"
    )
    args = parser.parse_args()

    model_settings = model_settings._replace(tokens_per_batch=args.tokens_per_batch)
    logging.debug(model_settings)

    logging.info("Loading token statistics")
    token_stats = dataprep2.TokenStatistics("model/all.tokenstats3.gz")

    logging.info("Loading embeddings")
    embeddings = dataprep2.CombinedEmbeddings(
        token_stats,
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.embedded_tokens_fraction
    )

    logging.info("Loading model")
    model = with_labels.model_with_labels(model_settings, embeddings)
    model.load_weights(args.model)

    logging.info("Starting server")
    server = Server(model, token_stats, embeddings, model_settings)
    server.serve_forever()

if __name__ == "__main__":
    main()
