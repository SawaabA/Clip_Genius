def find_individual_chars(image):
    height, widht, _ = image.shape
    height, widht
    boxes = pytesseract.image_to_boxes(image, config=config)
    for box in boxes.splitlines():
        box = box.split(" ")

        point1 = (int(box[1]), height - int(box[2]))
        point2 = (int(box[3]), height - int(box[4]))
        image = cv.rectangle(image, pt1=point1, pt2=point2, color=COLOR)
        print(box)
    return image


def find_words(image):
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    amount_of_boxes = len(data["text"])
    for i in range(amount_of_boxes):
        if float(data["conf"][i]) > 20:
            (x, y, w, h) = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            image = cv.rectangle(image, (x, y), (x + w, y + h), COLOR, thickness=2)
            image = cv.putText(
                image,
                data["text"][i],
                (x, y + h),
                cv.FONT_HERSHEY_DUPLEX,
                0.7,
                COLOR,
                thickness=2,
            )

    return image
