from queue import Queue
import cv2
import threading
from kafka import KafkaConsumer
from realtime_reid.pipeline import Pipeline

reid_pipeline = Pipeline()

# Create Kafka consumers for both topics
consumer_00 = KafkaConsumer(
    'topic_camera_00',
    bootstrap_servers=['localhost:9092']
)

consumer_01 = KafkaConsumer(
    'topic_camera_01',
    bootstrap_servers=['localhost:9092']
)

# Create a Queue to hold the processed images
processed_images = Queue()


def process_messages(consumer, consumer_name):
    for msg in consumer:
        # Process the message
        final_img = reid_pipeline.process(msg.value)

        # Add the processed image to the Queue
        processed_images.put((consumer_name, final_img))


# Start processing messages from both topics using threads
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

# Display the processed images in the main thread
while True:
    # Get the next processed image from the Queue
    consumer_name, final_img = processed_images.get()

    # Display the image
    cv2.imshow(consumer_name, final_img)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Wait for both threads to finish
thread_0.join()
thread_1.join()

# Closes all the frames
cv2.destroyAllWindows()
