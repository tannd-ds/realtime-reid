import os

import cv2
import numpy as np
import torch

from .classifier import PersonReID
from .feature_extraction import PersonDescriptor
from .person_detector import PersonDetector
from .visualization_utils import color, drawer

from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


class Pipeline:
    def __init__(self,
                detector: PersonDetector = PersonDetector(),
                descriptor: PersonDescriptor = PersonDetector(),
                classifier: PersonReID = PersonReID()) -> None:
        """Initialize the pipeline by creating the necessary objects."""

        self.detector = detector
        self.descriptor = descriptor
        self.classifier = classifier

        # Deep SORT
        max_cosine_distance = 0.2
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.nms_max_overlap = 1.0
        self.tracker = Tracker(metric)

        # Deep SORT
        max_cosine_distance = 0.2
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.nms_max_overlap = 1.0
        self.tracker = Tracker(metric)

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
        if not isinstance(msg, bytes) and not isinstance(msg, np.ndarray):
            raise TypeError(f"msg must be of type bytes or numpy array. Got {type(msg)}.")

        detected_data = self.detector.detect(msg)
        if len(detected_data) == 1:
            detected_data = detected_data[0]

        # Convert the image data to an array
        image_data = np.frombuffer(msg, dtype=np.uint8)
        final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # skip if there is no detection
        if len(detected_data.boxes) == 0:
            return final_img

        boxes = np.array([box.xyxy.squeeze().tolist() for box in detected_data.boxes])
        scores = np.array([box.conf.squeeze().tolist() for box in detected_data.boxes])

        features = torch.Tensor()
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)

            cropped_img = final_img[y_min:y_max, x_min:x_max, :]
            current_feature = self.descriptor.extract_feature(cropped_img).cpu()
            features = torch.cat((features, current_feature), dim=0)

            # A small solution for #3 (initial partial visibility)
            # This is a temporary solution, but it works beautifully.
            # Solution: Check if the person is fully visible
            offset = 2  # because the bbox is not (always) accurate
            lower_bound = ((x_min - offset) <= 0 or (y_min - offset) <= 0)
            upper_bound = (x_max + offset >= final_img.shape[1]
                           or (y_max + offset) >= final_img.shape[0])
            if lower_bound or upper_bound:
                current_id = -1
            else:
                current_id = self.classifier.identify(
                    target=current_feature,
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
            final_img = drawer.draw(
                final_img,
                str(current_id),
                (x_min, y_min),
                (x_max, y_max),
                color.create_unique_color(current_id)
            )

        # turn boxes from [x1, y1, x2, y2] to [x1, y1, w, h]
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        detections = [Detection(bbox, score, feature)
                      for bbox, score, feature
                      in zip(boxes, scores, features)]

        # Run non-maxima suppression.
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        # Draw bounding boxes and labels
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(int)
            label = f"{track.track_id}"
            final_img = drawer.draw(
                final_img,
                label,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color.create_unique_color(track.track_id)
            )

        if return_bytes:
            return cv2.imencode(
                '.jpg',
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )[1].tobytes()

        return final_img
