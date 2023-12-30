import argparse
from queue import Queue
import numpy as np
import cv2
import threading
from kafka import KafkaConsumer
from realtime_reid.pipeline import Pipeline


def parse_args():
    """
    Parse User's input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--bootstrap_servers",
                        type=str,
                        default="localhost:9092",
                        help="The Kafka Bootstrap Servers")
    parser.add_argument("-t1", "--topic-1",
                        type=str,
                        required=True,
                        help="The Topic Name")
    parser.add_argument("-t2", "--topic-2",
                        type=str,
                        default="NULL",
                        help="The Topic Name")
    parser.add_argument("-ir", "--is-raw",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True,
                        help="Whether the data is raw (not processed) or not")
    return parser.parse_args()


args = vars(parse_args())
BOOTSTRAP_SERVERS = args['bootstrap_servers']
TOPIC_1 = args['topic_1']
TOPIC_2 = args['topic_2']
IS_RAW = args['is_raw']

reid_pipeline = None
if IS_RAW:
    reid_pipeline = Pipeline()


# Create a Queue to hold the processed images
processed_images = Queue()


def process_messages(consumer: KafkaConsumer,
                     consumer_name: str):
    for msg in consumer:
        # Process the message
        final_img = np.frombuffer(msg.value, dtype=np.uint8)
        final_img = cv2.imdecode(final_img, cv2.IMREAD_COLOR)
        if IS_RAW:
            final_img = reid_pipeline.process(msg.value)

        # Add the processed image to the Queue
        processed_images.put((consumer_name, final_img))


def start_threads(consumer_00: KafkaConsumer,
                  consumer_01: KafkaConsumer):
    """Start processing messages from both topics using threads"""
    thread_0 = threading.Thread(
        target=process_messages,
        args=(consumer_00, "Camera 00")
    )
    thread_1 = threading.Thread(
        target=process_messages,
        args=(consumer_01, "Camera 01")
    )

    thread_0.start()
    thread_1.start()

    return thread_0, thread_1


def display_images():
    """Display the processed images in the main thread"""
    while True:
        # Get the next processed image and display it
        consumer_name, final_img = processed_images.get()
        cv2.imshow(consumer_name, final_img)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def main():
    # Create Kafka consumers for both topics
    consumer_00 = KafkaConsumer(
        TOPIC_1,
        bootstrap_servers=[BOOTSTRAP_SERVERS]
    )

    consumer_01 = KafkaConsumer(
        TOPIC_2,
        bootstrap_servers=[BOOTSTRAP_SERVERS]
    )

    thread_0, thread_1 = start_threads(consumer_00, consumer_01)
    display_images()

    # Wait for both threads to finish
    thread_0.join()
    thread_1.join()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
