import os

import cv2
import numpy as np

from .classifier import PersonReID
from .feature_extraction import PersonDescriptor
from .person_detector import PersonDetector
from .visualization_utils import color


class Pipeline:
    def __init__(self) -> None:
        """Initialize the pipeline by creating the necessary objects."""
        # Backbone models
        self.detector = PersonDetector('yolov8n.pt')
        self.descriptor = PersonDescriptor(use_pcb=True)
        self.classifier = PersonReID()

    def process(
        self,
        msg: bytes | np.ndarray,
        save_dir: str = None,
        return_bytes: bool = False
    ) -> np.ndarray | bytes:
        """
        Process the input message by detecting and identifying persons
        in the image.

        Parameters
        ----------
            msg (Message), bytes | np.ndarray, required
                The input message containing the image data.
            save_dir, str, default None
                The directory to save the detected images.
                Leave it empty if you don't want to save the images.
            return_bytes, bool, default False
                Whether to return the processed image as bytes.

        Returns
        -------
            Image: The processed image with bounding boxes and labels for
        detected persons.
        """
        if not isinstance(msg, bytes):
            raise TypeError("msg must be of type bytes.")

        detected_data = self.detector.detect_complex(msg)
        if len(detected_data) == 1:
            detected_data = detected_data[0]

        # Convert the image data to an array
        image_data = np.frombuffer(msg, dtype=np.uint8)
        final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        ids = []
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
                current_person = self.descriptor.extract_feature(cropped_img)
                current_id = self.classifier.identify(
                    target=current_person,
                    do_update=True
                )
            ids.append(current_id)

            # Save the cropped image before drawing the bounding box
            if save_dir is not None:
                save_filename = f"{len(os.listdir(save_dir))}_{current_id}"
                cv2.imwrite(
                    f"{save_dir}/{save_filename}.jpg",
                    cropped_img
                )

        for detected_box, current_id in zip(detected_data.boxes, ids):
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Draw bounding box and label
            label = f"{current_id}"
            cv2.rectangle(
                img=final_img,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=color.create_unique_color(current_id),
                thickness=2,
            )
            cv2.putText(
                img=final_img,
                text=label,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color.create_unique_color(current_id),
                thickness=2,
            )

        if return_bytes:
            return cv2.imencode(
                '.jpg',
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )[1].tobytes()

        return final_img
