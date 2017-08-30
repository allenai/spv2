#!/usr/bin/env python
# -*- coding: utf8 -*-

import http.server
import http.client
import tempfile
from sendfile import sendfile

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

        with tempfile.TemporaryFile(prefix="SPV2Server", suffix=".json", delete=True) as json_file:
            # get json from the dataprep server
            dataprep_conn = http.client.HTTPConnection("localhost", 8080, timeout=60)
            dataprep_conn.request("POST", self.path, body=self.rfile)
            with dataprep_conn.getresponse() as dataprep_response:
                if dataprep_response.status < 200 or dataprep_response.status >= 300:
                    raise ValueError("Error %d from dataprep server at %s" % (
                        dataprep_response.status,
                        dataprep_conn.host))
                sendfile(dataprep_response.fileno(), json_file.fileno(), None, 0)
            json_file.seek(0)

            # process the json



def main():
    server = http.server.HTTPServer(('', 8081), RequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
