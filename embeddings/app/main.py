import logging
from bottle import Bottle
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
from FeatureCloud.app.engine.app import app

import states

# Logger initialisieren
logging.basicConfig(level=logging.INFO)  # Setzt das Logging-Level
logger = logging.getLogger(__name__)
logger.info("Starting the FeatureCloud App...")

server = Bottle()

if __name__ == '__main__':
    logger.info("Registering app and mounting routes...")
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    logger.info("Starting the Bottle server on 0.0.0.0:5000")
    server.run(host='0.0.0.0', port=5000)
