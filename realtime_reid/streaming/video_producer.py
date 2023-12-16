import os
import cv2
import time
from kafka import KafkaProducer


class VideoProducer():
    def __init__(
            self,
            topic: str,
            interval: float | None = None,
            bootstrap_servers: str = 'localhost:9092'):

        self.INTERVAL = 0.2 if interval is None else interval
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.TOPIC = topic

    def encode_and_produce(self, frame, interval: int):
        # Convert image to jpg format
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to Kafka
        self.producer.send(self.TOPIC, buffer.tobytes())

        time.sleep(interval)

    def publish_from_video(self, source_path):
        # Open video file
        video = cv2.VideoCapture(source_path)

        while (video.isOpened()):
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
                print(
                    f"Publish from video {source} to topic {self.TOPIC}")
                self.publish_from_video(source)
            else:
                print(
                    f"Publish from folder {source} to topic {self.TOPIC}")
                self.publish_from_img_folder(source)

            print('Publish complete!')
        except KeyboardInterrupt:
            print("Publish stopped.")
