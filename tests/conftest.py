import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Producer, Consumer
import glob
import os


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../examples/avro_test"))


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def read_avro():
    files = glob.glob(os.path.join(EXAMPLES_PATH, "*.avro"))
    files.sort()
    nfiles = len(files)
    for f in files:
        with open(f, "rb") as fo:
            yield fo.read()


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
            return True
        except Exception as e:
            return False


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    print("Kafka", docker_ip)
    topics = ["test"]
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("kafka", 9094)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    config = {"bootstrap.servers": "localhost:9094"}
    producer = Producer(config)
    try:
        for topic in topics:
            for data in read_avro():
                producer.produce(topic, value=data)
                producer.flush()
            print(f"produced to {topic}")
    except Exception as e:
        print(f"failed to produce to topic {topic}: {e}")
    return server
