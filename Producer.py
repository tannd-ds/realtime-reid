import argparse
from realtime_reid.streaming import VideoProducer


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

    parser.add_argument("-i", "--interval",
                        type=float,
                        required=False,
                        help="The delay between each image.")
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
    interval = args['interval']

    producer = VideoProducer(topic, interval)
    producer.publish_video(camera)


if __name__ == '__main__':
    main()
