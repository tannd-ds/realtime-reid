import cv2
import numpy as np
from kafka import KafkaConsumer

consumer_00 = KafkaConsumer(
    'processed_topic',
    bootstrap_servers=['localhost:9092']
)

for msg in consumer_00:
    # print(msg.value)

    frame_np = np.frombuffer(msg.value, dtype=np.uint8)
    img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    cv2.imshow("Processed", img)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
