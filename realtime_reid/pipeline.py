import cv2
import numpy as np

from realtime_reid.person_detector import PersonDetector
from realtime_reid.feature_extraction import ResNetReID
from realtime_reid.classifier import PersonReID


class Pipeline:
    def __init__(self):
        """Initialize the pipeline by creating the necessary objects."""
        # Backbone models
        self.person_detector = PersonDetector()
        self.fe_model = ResNetReID()
        self.classifier = PersonReID()
        self.colors = [
            (193, 18,  31),
            (0,   175, 185),
            (0,   0,   255),
            (102, 155, 188),
            (0,   255, 0),
            (255, 255, 0),
            (9,   208, 2),
        ] * 1000

    def process(self, msg: bytes, return_bytes: str = False):
        """
        Process the input message by detecting and identifying persons
        in the image.

        Parameters
        ----------
            msg (Message): The input message containing the image data.

        Returns
        -------
            Image: The processed image with bounding boxes and labels for
        detected persons.
        """
        if not isinstance(msg, bytes):
            raise TypeError("msg must be of type bytes.")

        detected_data = self.person_detector.detect_complex(msg)
        if len(detected_data) == 1:
            detected_data = detected_data[0]

        # Convert the image data to an array
        image_data = np.frombuffer(msg, dtype=np.uint8)
        final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        for detected_box in detected_data.boxes:
            # detected_box.xyxy is a (1, 4) tensor
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            cropped_img = final_img[ymin:ymax, xmin:xmax, :]

            # A small solution for #3 (initial partial visibility)
            # This is a temporary solution, but it works beautifully.
            # Solution: Check if the person is fully visible
            offset = 2  # because the bbox is not (always) accurate
            lower_bound = ((xmin - offset) <= 0 or (ymin - offset) <= 0)
            upper_bound = (xmax + offset >= final_img.shape[1]
                           or (ymax + offset) >= final_img.shape[0])
            if lower_bound or upper_bound:
                current_id = -1
            else:
                current_person = self.fe_model.extract_feature(cropped_img)
                current_id = self.classifier.identify(
                    current_person,
                    update_embeddings=True
                )

            # Draw bounding box and label
            label = f" ID: {current_id}"
            cv2.rectangle(
                img=final_img,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=self.colors[current_id],
                thickness=2,
            )
            cv2.putText(
                img=final_img,
                text=label,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=self.colors[current_id],
                thickness=2,
            )

        if return_bytes:
            return cv2.imencode(
                '.jpg',
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )[1].tobytes()

        return final_img
