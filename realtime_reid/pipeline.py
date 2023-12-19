from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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
        # Style for bounding boxes and labels
        self.colors = ['red', 'green', 'blue', 'cyan', 'black'] * 1000
        self.font = ImageFont.truetype("./static/fonts/open_san.ttf", 30)

    def process(self, msg):
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
        detected_data = self.person_detector.detect_complex(msg.value)

        final_img = Image.open(BytesIO(msg.value))
        for detection in detected_data['result'].xyxy[0]:
            xmin, ymin, xmax, ymax = map(int, detection[:4])

            cropped_img = final_img.crop((xmin, ymin, xmax, ymax))
            cropped_img = np.array(cropped_img)

            # A small solution for #3 (initial partial visibility)
            # This is a temporary solution, but it works beautifully.
            # Solution: Check if the person is fully visible
            offset = 2  # because the bbox is not (always) accurate
            lower_bound = ((xmin - offset) <= 0 or (ymin - offset) <= 0)
            upper_bound = (xmax + offset >= final_img.width
                           or (ymax + offset) >= final_img.height)
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
            draw = ImageDraw.Draw(final_img)
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline=self.colors[current_id],
                width=4,
            )
            draw.text(
                xy=(xmin, ymin),
                text=label,
                fill=self.colors[current_id],
                font=self.font,
            )

        return final_img
