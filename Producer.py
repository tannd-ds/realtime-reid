import os
import argparse
import time
import cv2
from kafka import KafkaProducer
from realtime_reid.streaming import VideoProducer

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


def main():
    """
    Producer will publish to Kafka Server a video file given as a system arg. 
    Otherwise it will default by streaming webcam feed.
    """
    args = vars(parse_args())
    camera = args['camera']
    topic = args['topic']

    producer = VideoProducer(topic, camera)
    producer.publish_video(video_path=camera, topic=topic)


if __name__ == '__main__':
    main()
