import cv2
import numpy as np


def build_grid_mask(frame):
    """
    根据第一帧生成网格线 mask。
    白色区域表示网格线附近，黑色区域表示普通背景。
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.bitwise_or(h_lines, v_lines)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    grid_mask = cv2.dilate(grid, dilate_kernel, iterations=1)

    return grid_mask


def is_near_grid(grid_mask, point, radius=10):
    """
    判断某个点是否靠近网格线。
    """

    if grid_mask is None or point is None:
        return False

    x, y = point
    x = int(x)
    y = int(y)

    h, w = grid_mask.shape[:2]

    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = min(x + radius, w)
    y2 = min(y + radius, h)

    if x2 <= x1 or y2 <= y1:
        return False

    patch = grid_mask[y1:y2, x1:x2]

    return np.count_nonzero(patch) > 0