def get_score_value(frame, cords):
    x1, y1, width1, height1 = cords[0]
    cropped_image_1 = frame[y1 : y1 + height1, x1 : x1 + width1]

    x2, y2, width2, height2 = cords[1]
    cropped_image_2 = frame[y2 : y2 + height2, x2 : x2 + width2]

    height = min(cropped_image_1.shape[0], cropped_image_2.shape[0])
    widht = min(cropped_image_1.shape[1], cropped_image_2.shape[1])

    cropped_image_1 = cv.resize(cropped_image_1, (height, widht))
    cropped_image_2 = cv.resize(cropped_image_2, (height, widht))

    combined = np.hstack((cropped_image_1, cropped_image_2))

    gray = cv.cvtColor(combined, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    binary = cv.resize(binary, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)

    custom_config = r"--oem 3 --psm 10"
    char = pytesseract.image_to_string(binary, config=custom_config)
    char = char.strip()
    print(char)
    return combined
