import argparse
from queue import Queue
import numpy as np
import cv2
import threading
from kafka import KafkaConsumer
from realtime_reid.pipeline import Pipeline

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC_2 = "NULL"  # No second topic by default
DEFAULT_APPLY_REID = False  # Do not apply reid by default


def parse_args():
    """Parse User's input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--bootstrap-servers",
                        type=str,
                        default="localhost:9092",
                        help="The address of the Kafka bootstrap"
                        "servers in the format 'host:port'")
    parser.add_argument("-t", "--topic", "--topic-1",
                        type=str,
                        required=True,
                        help="The name of the first kafka topic")
    parser.add_argument("-t2", "--topic-2",
                        type=str,
                        default="NULL",
                        help="The name of the second kafka topic (optional)")
    parser.add_argument("-r", "--reid",
                        type=str,
                        choices=["y", "n", "spark"],
                        default="n",
                        help="Set this 'y' if you want to apply reid on the"
                        "images, 'spark' if you want to run the spark and")

    # misc
    parser.add_argument("-s", "--save-dir",
                        type=str,
                        default=None,
                        help="The directory to save the detected images."
                        "Leave it empty if you don't want to save the images")
    return parser.parse_args()


args = vars(parse_args())
BOOTSTRAP_SERVERS = args['bootstrap_servers']
TOPIC_1 = args['topic']
TOPIC_2 = args['topic_2']
INTEGRATE_SPARK = (args['reid'] == "spark")
APPLY_REID = (args['reid'] == "y" or INTEGRATE_SPARK)

reid_pipeline = None
if APPLY_REID and not INTEGRATE_SPARK:
    reid_pipeline = Pipeline()


# Create a Queue to hold the processed images
processed_images = Queue()


def process_messages(consumer: KafkaConsumer,
                     consumer_name: str):
    for msg in consumer:
        # Process the message
        final_img = np.frombuffer(msg.value, dtype=np.uint8)
        final_img = cv2.imdecode(final_img, cv2.IMREAD_COLOR)
        if APPLY_REID and not INTEGRATE_SPARK:
            final_img = reid_pipeline.process(msg.value, save_dir=args['save_dir'])

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
    # Spark streaming thread
    if INTEGRATE_SPARK:
        from streaming.spark_services.spark_streaming import start_spark
        thread = threading.Thread(target=start_spark)
        thread.start()

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
