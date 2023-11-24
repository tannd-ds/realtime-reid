import argparse
from flask import Flask, Response, render_template
from kafka import KafkaConsumer
from PIL import Image, ImageDraw
from io import BytesIO
import torch
import torchvision.transforms as transforms


# Load YOLOv5 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Make YOLO only detect "person" (Class index 0 in COCO Dataset)
yolo_detector.classes = [0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", "--topic0", 
                        type=str, 
                        default="topic_camera_00",
                        help="The Topic Name for Camera 0.")
    parser.add_argument("-t2", "--topic1", 
                        type=str, 
                        default="topic_camera_01",
                        help="The Topic Name for Camera 1.")
    parser.add_argument("-p", "--port", 
                        type=int, 
                        default=5000,
                        help="Server port.")
    args = parser.parse_args()
    return args

# Fire up the Kafka Consumers
args = vars(parse_args())
topic0 = args['topic0']
topic1 = args['topic1']
port = args['port']

consumer1 = KafkaConsumer(
    topic0, 
    bootstrap_servers=['localhost:9092'])

consumer2 = KafkaConsumer(
    topic1, 
    bootstrap_servers=['localhost:9092'])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera1', methods=['GET'])
def camera1():
    return Response(
        get_video_stream(consumer1), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera2', methods=['GET'])
def camera2():
    return Response(
        get_video_stream(consumer2), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_human(input_image):
    # Convert the received image bytes to a PIL Image
    image = Image.open(BytesIO(input_image))

    # Perform YOLOv5 object detection
    results = yolo_detector([image])

    results.crop(save=True)

    results.render()

    # Draw bounding boxes around detected objects
    for detection in results.xyxy[0]:
        # if int(detection[-1]) == 0:  # Check if the detected object is a person (class 0 for COCO dataset)
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            label = f"{yolo_detector.names[int(detection[-1])]} {detection[-2]:.2f}"
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin), label, fill="red")

    # Convert the modified image back to bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    return buffered

def get_video_stream(consumer):
    """
    Here is where we receive streamed images from the Kafka Server and convert 
    them to a Flask-readable format.
    """
    for msg in consumer:
        buffered = detect_human(msg.value)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffered.getvalue() + b'\r\n\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
