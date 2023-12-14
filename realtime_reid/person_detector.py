import torch
from io import BytesIO
from PIL import Image, ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonDetector():
    def __init__(self):
        # Load YOLOv5 Model
        self.yolo = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path='checkpoints/yolov5s.pt',
        )

        # Only detect "person" (Class index 0 in COCO Dataset)
        self.yolo.classes = [0]

    def detect_human(self, input_image):
        """
        Return Images with detected boxes for each person.

        Parameters
        ----------
        input_image: byte
            an Image stored in `bytes` type that need to be detected.

        Returns
        -------
        The result Image after drawing detected boxes, stored in `bytes`.
        """
        # Convert the received image bytes to a PIL Image
        image = Image.open(BytesIO(input_image))

        # Perform YOLOv5 object detection
        results = self.yolo([image])

        # Crop & Save detected people for Re-ID task
        results.crop(save=True)

        results.render()

        # Draw bounding boxes around detected objects
        for detection in results.xyxy[0]:
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            label = f"{self.yolo.names[int(detection[-1])]} {detection[-2]:.2f}"
            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin), label, fill="red")

        # Convert the modified image back to bytes
        buffered = BytesIO()
        image.save(buffered, format="JPEG")

        return buffered
