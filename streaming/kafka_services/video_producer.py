import os
import time

import cv2
from kafka import KafkaProducer


class VideoProducer:
    def __init__(
            self,
            topic: str,
            interval: float,
            bootstrap_servers: str = 'localhost:9092'):

        self.INTERVAL = interval
        # Incase user input FPS instead of interval
        if self.INTERVAL > 1:
            self.INTERVAL = 1 / self.INTERVAL

        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.TOPIC = topic

    def encode_and_produce(self, frame, interval: float):
        frame = self.process_frame(frame)

        # Convert image to jpg format
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to Kafka
        self.producer.send(self.TOPIC, buffer.tobytes())

        time.sleep(interval)

    def publish_from_video(self, source_path):
        # Open video file
        video = cv2.VideoCapture(source_path)

        # Set default interval for video to video FPS
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / video.get(cv2.CAP_PROP_FPS)

        while video.isOpened():
            success, frame = video.read()

            # Ensure file was read successfully
            if not success:
                print("bad read!")
                break
            self.encode_and_produce(frame, self.INTERVAL)
        video.release()
        return True

    def publish_from_img_folder(self, source_path):
        # Open folder
        image_files = [f for f in os.listdir(source_path)
                       if f.endswith(('.jpg', '.png'))]

        # Set default interval for image folder to 12 FPS
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / 12

        for img in image_files:
            image_path = os.path.join(source_path, img)
            frame = cv2.imread(image_path)

            self.encode_and_produce(frame, self.INTERVAL)
        return True

    def publish_video(self, source: str):
        """
        Publish given video file to `self.topic`.
        There are 2 possible `video_path` input, a link to a video
        (a file) or a path to a folder that contains of images.

        Parameters
        ----------
        `source`: str
            path to video file (camera demo)
        """
        try:
            if os.path.isfile(source):
                print(f"Publish from video {source} to topic {self.TOPIC}")
                self.publish_from_video(source)
            else:
                print(f"Publish from folder {source} to topic {self.TOPIC}")
                self.publish_from_img_folder(source)

            print('Publish complete!')
        except KeyboardInterrupt:
            print("Publish stopped.")

    @staticmethod
    def process_frame(frame):
        # Image with fixed size, reserve aspect ratio
        original_ratio = frame.shape[1] / frame.shape[0]
        width = 640
        height = int(width / original_ratio)
        frame = cv2.resize(frame, (width, height))

        return frame
