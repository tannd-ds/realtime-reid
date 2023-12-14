import os
import argparse
import time
import cv2
from kafka import KafkaProducer

INTERVAL = 0.2


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


def publish_video(video_path: str, topic):
    """
    Publish given video file to a specified Kafka topic. 
    There are 2 possible `video_path` input, a link to a video (end with ".mp4")
    or a path to a folder full of images.

    Kafka Server is expected to be running on the localhost. Not partitioned.

    Parameters
    ----------
    `video_path`: str
        path to video file (camera demo)
    `topic`: str
        the topic to be published to.
    """

    def encode_and_produce(frame, producer: KafkaProducer, interval: int):
        # Convert image to jpg format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to Kafka
        producer.send(topic, buffer.tobytes())

        time.sleep(interval)

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    if '.mp4' in video_path:
        # Open file
        video = cv2.VideoCapture(video_path)
        print('publishing video...')

        while (video.isOpened()):
            success, frame = video.read()

            # Ensure file was read successfully
            if not success:
                print("bad read!")
                break
            encode_and_produce(
                frame=frame, producer=producer, interval=INTERVAL)
        video.release()
    else:
        # Open folder
        image_files = [f for f in os.listdir(video_path)
                       if f.endswith(('.jpg', '.png'))]
        print('publishing video...')

        for img in image_files:
            image_path = os.path.join(video_path, img)
            frame = cv2.imread(image_path)

            encode_and_produce(
                frame=frame,
                producer=producer,
                interval=INTERVAL
            )

    print('publish complete')


def main():
    """
    Producer will publish to Kafka Server a video file given as a system arg. 
    Otherwise it will default by streaming webcam feed.
    """
    args = vars(parse_args())
    camera = args['camera']
    topic = args['topic']

    publish_video(video_path=camera, topic=topic)


if __name__ == '__main__':
    main()
