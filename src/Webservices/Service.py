from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import cgi
from src.ImageProcessing.Defacer import defacer

class WebServerHandler(BaseHTTPRequestHandler):
    """
    This class handles any incoming request from the browser
    """

    store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'tmp' )
    print(store_path)

    def do_POST(self):

        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == "multipart/form-data":
            fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST',"CONTENT_TYPE": self.headers['Content-Type']})

            uploaded_file = fs.list[0]
            filename = uploaded_file.disposition_options['filename']

            store_path = os.path.join(self.store_path, filename)

            if uploaded_file is not None:
                with open(store_path, "wb") as fh:
                    fh.write(uploaded_file.file.read())
            s = b"uploaded, thanks"

            file_uploaded = defacer(store_path)
            fl_size = self.file_size(file_uploaded)
            file_s = open(file_uploaded, 'rb')
            data = file_s.read()
            file_s.close()
            self.respond(response=data, size=fl_size, content_type="multipart/form-data", status=200)

        else:
            resp = b"failed to upload %s, not multipart/form-data"
            length = len(resp)
            self.respond(response=resp, size=length, content_type="text/html", status=400)

    def respond(self, response, size, content_type, status):
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.send_header("Content-length", size)
        self.end_headers()
        self.wfile.write(response)

    def file_size(self, filepath):
        stat_info = os.stat(filepath)
        return stat_info.st_size


if __name__ == '__main__':

    PORT_SERVER = 5000
    SERVER_ADDRESS = ('127.0.0.1', PORT_SERVER)

    try:
        # Create a web server and define the handler to manage the incoming request
        ServerClass = HTTPServer
        httpd = ServerClass(SERVER_ADDRESS, WebServerHandler)

        sa = httpd.socket.getsockname()
        print("Serving HTTP on", sa[0], "port", sa[1], "...")

        # Wait forever for incoming http requests
        httpd.serve_forever()

    except KeyboardInterrupt:
        print('received, shutting down the web server')
        httpd.socket.close()