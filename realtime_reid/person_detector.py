from io import BytesIO
from PIL import Image
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

    def detect_human(self, input_image: bytes, save_detection: bool = False):
        """
        Return Images with detected boxes for each person.

        Parameters
        ----------
        input_image: bytes
            an Image stored in `bytes` type that need to be detected.
        save_detection: bool, defaut False
            wheater to save image and detected people into disk.

        Returns
        -------
        The result Image after drawing detected boxes, stored in `bytes`.
        """
        # Convert the received image bytes to a PIL Image
        image = Image.open(BytesIO(input_image))

        # Perform YOLOv5 object detection
        results = self.yolo([image])

        # Crop & Save detected people for Re-ID task
        results.crop(save=save_detection)
        drawed_image = Image.fromarray(results.render()[0])

        # Convert the modified image back to bytes
        buffered = BytesIO()
        drawed_image.save(buffered, format="JPEG")

        return buffered

    def detect_complex(
        self,
        input_bytes: bytes,
        save_detection: bool = False
    ):
        """
        Return (alot of) data store of a detected image, save in `dict`.

        Parameters
        ----------
        input_bytes: byte
            an Image stored in `bytes` type that need to be detected.

        save_detection: bool, defaut False
            wheater to save image and detected people into disk.

        Returns
        -------
        The result Image after drawing detected boxes, stored in `bytes`.
        """
        # Convert the received image bytes to a PIL Image
        image = Image.open(BytesIO(input_bytes))

        # Perform YOLOv5 object detection
        results = self.yolo([image])

        # Crop & Save detected people for Re-ID task
        detected_ppl = results.crop(save=save_detection)

        return {
            'result': results,
            'detected_ppl': detected_ppl,
        }
