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
    allowed_paths = {"/v1/tar", "/v1/targz", "/v1/zip", "/v1/pdf", "/v1/urls"}

    def do_GET(self):
        if self.path in self.allowed_paths:
            self.send_error(405)
        else:
            self.send_error(404)

    def do_PUT(self):
        self.send_error(405)

    def do_DELETE(self):
        self.send_error(405)

    def do_PATCH(self):
        self.send_error(405)

    def do_POST(self):
        if self.path not in self.allowed_paths:
            self.send_error(404)
            return

        with tempfile.TemporaryDirectory(prefix="SPV2Server-") as temp_dir:
            # read input
            input_size = int(self.headers["Content-Length"])
            input_file_name = os.path.join(temp_dir, "input")
            with open(input_file_name, "wb") as input_file:
                _send_all(self.rfile, input_file, input_size)

            # get json from the dataprep server
            json_file_name = os.path.join(temp_dir, "tokens.json")
            with open(json_file_name, "wb") as json_file, open(input_file_name, "rb") as input_file:
                dataprep_conn = http.client.HTTPConnection("localhost", 8080, timeout=60)
                dataprep_conn.request("POST", self.path, body=input_file)
                with dataprep_conn.getresponse() as dataprep_response:
                    if dataprep_response.status < 200 or dataprep_response.status >= 300:
                        raise ValueError("Error %d from dataprep server at %s" % (
                            dataprep_response.status,
                            dataprep_conn.host))
                    _send_all(dataprep_response, json_file)
            os.remove(input_file_name)

            # make unlabeled tokens file
            unlabeled_tokens_file_name = os.path.join(temp_dir, "unlabeled-tokens.h5")
            dataprep2.make_unlabeled_tokens_file(json_file_name, unlabeled_tokens_file_name)
            os.remove(json_file_name)

            # make featurized file
            with h5py.File(unlabeled_tokens_file_name, "r") as unlabeled_tokens_file:
                featurized_tokens_file_name = os.path.join(temp_dir, "featurized-tokens.h5")
                dataprep2.make_featurized_tokens_file(
                    featurized_tokens_file_name,
                    unlabeled_tokens_file,
                    self.server.token_stats,
                    self.server.embeddings,
                    dataprep2.VisionOutput(None),   # TODO: put in real vision output
                    self.server.model_settings
                )
                # We don't delete the unlabeled file here because the featurized one contains references
                # to it.

            with h5py.File(featurized_tokens_file_name) as featurized_tokens_file:
                def get_docs():
                    return dataprep2.documents_for_featurized_tokens(
                        featurized_tokens_file,
                        include_labels=False)
                results = with_labels.run_model(
                    self.server.model,
                    self.server.model_settings,
                    get_docs)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                response_body = codecs.getwriter("UTF-8")(self.wfile, "UTF-8")
                for doc, title, authors in results:
                    result_json = {
                        "docId": doc.doc_id,
                        "title": title,
                        "authors": authors
                    }
                    json.dump(result_json, response_body)
                    response_body.write("\n")
                response_body.reset()


class Server(http.server.HTTPServer):
    def __init__(self, model, token_stats, embeddings, model_settings):
        super(Server, self).__init__(('', 8081), RequestHandler)

        self.model = model
        self.token_stats = token_stats
        self.embeddings = embeddings
        self.model_settings = model_settings


def main():
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
        required=True,
        help="filename of existing model"
    )
    parser.add_argument(
        "--tokenstats",
        type=str,
        required=True,
        help="filename of the tokenstats file"
    )
    parser.add_argument(
        "--glove-vectors",
        type=str,
        default=model_settings.glove_vectors,
        help="file containing the GloVe vectors"
    )
    args = parser.parse_args()

    model_settings = model_settings._replace(tokens_per_batch=args.tokens_per_batch)
    model_settings = model_settings._replace(glove_vectors=args.glove_vectors)
    logging.debug(model_settings)

    logging.info("Loading token statistics")
    token_stats = dataprep2.TokenStatistics(args.tokenstats)

    logging.info("Loading embeddings")
    embeddings = dataprep2.CombinedEmbeddings(
        token_stats,
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.minimum_token_frequency
    )

    logging.info("Loading model")
    model = with_labels.model_with_labels(model_settings, embeddings)
    model.summary()
    model.load_weights(args.model)

    logging.info("Starting server")
    server = Server(model, token_stats, embeddings, model_settings)
    server.serve_forever()

if __name__ == "__main__":
    main()
