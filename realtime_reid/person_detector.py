import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonDetector():
    def __init__(self, model_path: str = 'yolov5n.pt'):
        # Load YOLOv8 Model
        self.yolo = YOLO(model_path)

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

        # Perform object detection
        results = self.yolo(image, classes=[0])
        return results
