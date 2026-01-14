# ruff: noqa: E501
from bottle import Bottle

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server

from FeatureCloud.app.engine.app import app

import sys

import states

server = Bottle()

if __name__ == "__main__":
    try:
        app.register()
        server.mount("/api", api_server)
        server.mount("/web", web_server)
        server.run(host="localhost", port=5000)
    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(1)
