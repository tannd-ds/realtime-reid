import cv2
import numpy as np
import torch

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

    def detect_complex(
        self,
        input_bytes: bytes,
    ):
        """
        Return (alot of) data store of a detected image, save in `dict`.

        Parameters
        ----------
        input_bytes: byte
            an Image stored in `bytes` type that need to be detected.

        Returns
        -------
        The result Image after drawing detected boxes, stored in `bytes`.
        """
        # Convert the received image bytes to a PIL Image
        image = cv2.imdecode(
            np.frombuffer(input_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Perform YOLOv5 object detection
        results = self.yolo([image])
        return {
            'result': results,
        }
