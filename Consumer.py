import argparse
from flask import Flask, Response, render_template
from kafka import KafkaConsumer
import torch
from realtime_reid import PersonDetector

import findspark
from pyspark.sql import SparkSession


def parse_args():
    """
    Parse User's input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bootstrap_server",
                        type=str,
                        default="localhost:9092",
                        help="Kafka Bootstrap Server")
    parser.add_argument("-t0", "--topic_0",
                        type=str,
                        default="topic_camera_00",
                        help="The Topic Name for Camera 0.")
    parser.add_argument("-t1", "--topic_1",
                        type=str,
                        default="topic_camera_01",
                        help="The Topic Name for Camera 1.")
    parser.add_argument("-p", "--port",
                        type=int,
                        default=5000,
                        help="Port to run the Application.")
    return parser.parse_args()


# User input arguments Constants
args = vars(parse_args())
BOOSTRAP_SERVER = args['bootstrap_server']
TOPIC_0 = args['topic_0']
TOPIC_1 = args['topic_1']
PORT = args['port']
# Other setup Constants
SCALA_VERSION = '2.12'
SPARK_VERSION = '3.5.0'
KAFKA_VERSION = '3.6.0'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
    f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
]
# Fire up Spark
findspark.init()
spark = SparkSession.builder \
    .master('local') \
    .appName("person-reid") \
    .config("spark.jars.packages", ",".join(packages)) \
    .getOrCreate()

detector = PersonDetector()

# Fire up the Kafka Consumers
consumer1 = KafkaConsumer(
    TOPIC_0,
    bootstrap_servers=[BOOSTRAP_SERVER])

consumer2 = KafkaConsumer(
    TOPIC_1,
    bootstrap_servers=[BOOSTRAP_SERVER])

app = Flask(__name__)


@app.route('/')
def index():
    """
    Main Route that displays the Dashboard for our Cameras.
    """
    return render_template('index.html')


@app.route('/camera1', methods=['GET'])
def camera1():
    """
    Route that contains the data from Camera 1.
    """
    return Response(
        get_video_stream(consumer1),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera2', methods=['GET'])
def camera2():
    """
    Route that contains the data from Camera 2.
    """
    return Response(
        get_video_stream(consumer2),
        mimetype='multipart/x-mixed-replace; boundary=frame')


def get_video_stream(consumer):
    """
    Here is where we receive streamed images from the Kafka Server and convert 
    them to a Flask-readable format.
    """
    for msg in consumer:
        buffered = detector.detect_human(msg.value)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffered.getvalue() + b'\r\n\r\n')


def main():
    app.run(host='0.0.0.0', port=PORT, debug=True)


if __name__ == "__main__":
    main()
