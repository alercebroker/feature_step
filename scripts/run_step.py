import os
import sys
import logging

from settings import CONSUMER_CONFIG, STEP_CONFIG

from features import FeaturesComputer
from apf.consumers import KafkaConsumer as Consumer


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

level = logging.INFO

debug = os.getenv("LOGGING_DEBUG")

if debug in ("True", "true", "1"):
    level = logging.DEBUG
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

consumer = Consumer(config=CONSUMER_CONFIG)
step = FeaturesComputer(consumer, config=STEP_CONFIG, level=level)
step.start()
