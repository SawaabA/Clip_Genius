import cv2 as cv
import pytesseract
import numpy as np
from pytesseract import Output

# Constants
CONFIG = r"--psm 11 --oem 3"
COLOR = (0, 255, 0)
CONFIDENCE_THRESHOLD = 40


def find_scores(image: np.ndarray, confidence_threshold: int = CONFIDENCE_THRESHOLD):
    annotated_image = image.copy()

    # Perform OCR
    data = pytesseract.image_to_data(
        annotated_image, config=CONFIG, output_type=Output.DICT
    )
    cords = []
    for i in range(len(data["text"])):
        text, conf = data["text"][i], float(data["conf"][i])
        if text.strip() == "0" and conf > confidence_threshold:
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )

            cords.append(
                (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
            )

    return cords
