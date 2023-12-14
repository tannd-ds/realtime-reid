import os
import cv2
import time
from kafka import KafkaProducer


class VideoProducer():
    def __init__(self, topic: str, source: str, bootstrap_servers: str = 'localhost:9092',):
        # Const
        self.INTERVAL = 0.2

        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.TOPIC = topic
        self.source = source

    def encode_and_produce(self, frame, interval: int):
        # Convert image to jpg format
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to Kafka
        self.producer.send(self.TOPIC, buffer.tobytes())

        time.sleep(interval)

    def publish_from_video(self, video_path):
        # Open file
        video = cv2.VideoCapture(video_path)
        print('publishing video...')

        while (video.isOpened()):
            success, frame = video.read()

            # Ensure file was read successfully
            if not success:
                print("bad read!")
                break
            self.encode_and_produce(frame, self.INTERVAL)
        video.release()
        return True

    def publish_from_img_folder(self, folder_path):
        # Open folder
        image_files = [f for f in os.listdir(folder_path)
                       if f.endswith(('.jpg', '.png'))]
        print('publishing video...')

        for img in image_files:
            image_path = os.path.join(folder_path, img)
            frame = cv2.imread(image_path)

            self.encode_and_produce(frame, self.INTERVAL)
        return True

    def publish_video(self, video_path: str, topic):
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

        if '.mp4' in video_path:
            self.publish_from_video(video_path)
        else:
            self.publish_from_img_folder(video_path)
        print('publish complete')
