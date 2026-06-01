import cv2
import numpy as np
import math


def detect_drops(frame, search_center=None, search_radius=80):

    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]

    offset_x = 0
    offset_y = 0
    roi = frame

    if search_center is not None:

        cx, cy = map(int, search_center)

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return []

        x1 = max(cx - search_radius, 0)
        y1 = max(cy - search_radius, 0)
        x2 = min(cx + search_radius, w)
        y2 = min(cy + search_radius, h)

        if x2 <= x1 or y2 <= y1:
            return []

        roi = frame[y1:y2, x1:x2]

        offset_x = x1
        offset_y = y1

    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # LoG增强亮点
    log = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    log = np.abs(log)

    log = cv2.normalize(
        log,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    mean = np.mean(log)
    std = np.std(log)

    thresh = max(20, mean + 2.0 * std)

    _, binary = cv2.threshold(
        log,
        thresh,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    drops = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < 4:
            continue

        if area > 300:
            continue

        perimeter = cv2.arcLength(c, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * math.pi * area /
            (perimeter * perimeter)
        )

        if circularity < 0.25:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)

        if r < 2:
            continue

        if r > 12:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)

        cv2.drawContours(
            mask,
            [c],
            -1,
            255,
            -1
        )

        brightness = cv2.mean(
            gray,
            mask=mask
        )[0]

        score = (
            brightness * 2.0 +
            circularity * 100 -
            abs(r - 4) * 5
        )

        drops.append(
            (
                int(x + offset_x),
                int(y + offset_y),
                int(max(r, 3)),
                score
            )
        )

    drops.sort(
        key=lambda d: d[3],
        reverse=True
    )

    return [
        (x, y, r)
        for x, y, r, _
        in drops
    ]