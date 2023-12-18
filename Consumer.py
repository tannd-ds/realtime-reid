import argparse
from io import BytesIO
from flask import Flask, Response, render_template
from kafka import KafkaConsumer
from realtime_reid import Pipeline


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

reid_pipeline = Pipeline()

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
    """Route that contains the data from Camera 1."""
    return Response(
        get_video_stream(consumer1),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera2', methods=['GET'])
def camera2():
    """Route that contains the data from Camera 2."""
    return Response(
        get_video_stream(consumer2),
        mimetype='multipart/x-mixed-replace; boundary=frame')


def get_video_stream(consumer):
    """
    Generates a video stream by processing messages from a consumer.

    Parameters
    ----------
    consumer : object
        The consumer object that provides messages.

    Yields
    ------
    bytes
        A frame of the video stream in the form of a JPEG image.
    """
    for msg in consumer:

        final_img = reid_pipeline.process(msg)

        buffered = BytesIO()
        final_img.save(buffered, format='jpeg')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffered.getvalue() + b'\r\n\r\n')


def main():
    app.run(host='0.0.0.0', port=PORT, debug=True)


if __name__ == "__main__":
    main()
