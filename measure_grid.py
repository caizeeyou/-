import cv2
import numpy as np


def detect_horizontal_grid_lines(frame):
    """
    自动检测视频帧中的横向网格线 y 坐标。
    返回值：
        grid_y_pixels: [y1, y2, y3, ...]
    """

    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 轻微平滑，减少屏幕噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 提取亮线
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = binary.shape[:2]

    # 用横向结构元素提取横线
    kernel_width = max(60, w // 10)
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        h_kernel
    )

    # 稍微膨胀，让横线更连续
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3)
    )

    horizontal_lines = cv2.dilate(
        horizontal_lines,
        dilate_kernel,
        iterations=1
    )

    # 对每一行统计白色像素数量
    projection = np.sum(horizontal_lines > 0, axis=1).astype(np.float32)

    # 平滑投影曲线
    smooth_kernel = np.ones(9) / 9
    projection_smooth = np.convolve(
        projection,
        smooth_kernel,
        mode="same"
    )

    if projection_smooth.max() <= 0:
        return []

    # 自动阈值，找横线峰值
    threshold = max(
        projection_smooth.mean() + projection_smooth.std(),
        projection_smooth.max() * 0.25
    )

    line_regions = projection_smooth > threshold

    grid_y_pixels = []
    start = None

    for i, is_line in enumerate(line_regions):
        if is_line and start is None:
            start = i

        if not is_line and start is not None:
            end = i - 1

            if end - start >= 1:
                region = projection_smooth[start:end + 1]
                ys = np.arange(start, end + 1)

                # 加权平均，得到横线中心 y 坐标
                center_y = int(np.average(ys, weights=region))
                grid_y_pixels.append(center_y)

            start = None

    # 处理最后一段
    if start is not None:
        end = len(line_regions) - 1
        region = projection_smooth[start:end + 1]
        ys = np.arange(start, end + 1)
        center_y = int(np.average(ys, weights=region))
        grid_y_pixels.append(center_y)

    # 去掉过近的重复线
    grid_y_pixels = sorted(grid_y_pixels)

    merged = []

    for y in grid_y_pixels:
        if len(merged) == 0:
            merged.append(y)
        else:
            if abs(y - merged[-1]) > 10:
                merged.append(y)

    return merged


def draw_detected_grid_lines(frame, grid_y_pixels):
    """
    用于调试：把自动检测到的横线画出来。
    """

    display = frame.copy()

    for y in grid_y_pixels:
        cv2.line(
            display,
            (0, int(y)),
            (display.shape[1], int(y)),
            (0, 255, 255),
            1
        )

        cv2.putText(
            display,
            str(int(y)),
            (20, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    return display