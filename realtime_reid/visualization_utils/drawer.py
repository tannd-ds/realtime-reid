import cv2
import numpy as np

def draw(
        target: cv2.Mat | np.ndarray,
        label: str,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2
) -> cv2.Mat | np.ndarray:
    cv2.rectangle(
        img=target,
        pt1=pt1,
        pt2=pt2,
        color=color,
        thickness=thickness,
    )
    cv2.putText(
        img=target,
        text=label,
        org=pt1,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=thickness
    )

    return target