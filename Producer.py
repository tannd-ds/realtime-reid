import argparse
import time
import cv2
from kafka import KafkaProducer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic",
                        type=str,
                        required=True,
                        help="The Topic Name.")

    parser.add_argument("-c", "--camera",
                        type=str,
                        required=False,
                        help="Path to the camera demo video.")
    args = parser.parse_args()
    return args


def publish_video(video_file, topic):
    """
    Publish given video file to a specified Kafka topic. 
    Kafka Server is expected to be running on the localhost. Not partitioned.

    ----------
    Parameters:
    - `video_file`: str, path to video file (camera demo)
    - `topic`: str, the topic to be published to.
    """

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    # Open file
    video = cv2.VideoCapture(video_file)

    print('publishing video...')

    while (video.isOpened()):
        success, frame = video.read()

        # Ensure file was read successfully
        if not success:
            print("bad read!")
            break

        # Convert image to png
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to kafka
        producer.send(topic, buffer.tobytes())

        time.sleep(0.05)
    video.release()
    print('publish complete')


def main():
    """
    Producer will publish to Kafka Server a video file given as a system arg. 
    Otherwise it will default by streaming webcam feed.
    """
    args = vars(parse_args())
    camera = args['camera']
    topic = args['topic']

    publish_video(video_file=camera, topic=topic)


if __name__ == '__main__':
    main()
