import os
import sys
import logging
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)
from settings import CONSUMER_CONFIG, STEP_CONFIG

level = logging.INFO
if os.getenv("LOGGING_DEBUG"):
    level=logging.DEBUG
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=level,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

from features import FeaturesComputer
from apf.consumers import KafkaConsumer as Consumer

consumer = Consumer(config=CONSUMER_CONFIG)
step = FeaturesComputer(consumer,config=STEP_CONFIG,level=level)
step.start()
