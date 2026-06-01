import numpy as np


def calculate_velocity_with_grid(
    trajectory,
    fps,
    grid_y_pixels,
    grid_distance_mm=0.25,
    fit_points=25
):
    """
    只使用真实检测点 measured=True 做速度拟合。
    """

    if fps <= 0:
        return 0.0

    if grid_y_pixels is None or len(grid_y_pixels) < 3:
        return 0.0

    measured_points = [p for p in trajectory if len(p) >= 3 and p[2]]
    if len(measured_points) < 8:
        return 0.0

    measured_points = measured_points[-fit_points:]

    grid_y_pixels = np.array(sorted(grid_y_pixels), dtype=float)
    gaps = np.diff(grid_y_pixels)

    if len(gaps) == 0:
        return 0.0

    median_gap = np.median(gaps)
    if median_gap <= 0:
        return 0.0

    grid_indices = np.round(
        (grid_y_pixels - grid_y_pixels[0]) / median_gap
    ).astype(int)

    unique_indices = []
    unique_pixels = []

    for idx, y in zip(grid_indices, grid_y_pixels):
        if idx not in unique_indices:
            unique_indices.append(idx)
            unique_pixels.append(y)

    unique_indices = np.array(unique_indices, dtype=float)
    unique_pixels = np.array(unique_pixels, dtype=float)

    if len(unique_pixels) < 3:
        return 0.0

    grid_y_mm = unique_indices * grid_distance_mm

    y_pixels = np.array([p[1] for p in measured_points], dtype=float)
    y_mm = np.interp(y_pixels, unique_pixels, grid_y_mm)

    t_values = np.arange(len(y_mm), dtype=float) / fps

    if len(y_mm) >= 7:
        median_y = np.median(y_mm)
        mad = np.median(np.abs(y_mm - median_y))
        if mad > 1e-9:
            mask = np.abs(y_mm - median_y) <= 3.5 * mad
            if np.count_nonzero(mask) >= 5:
                y_mm = y_mm[mask]
                t_values = t_values[mask]

    if len(y_mm) < 5:
        return 0.0

    velocity_mm_per_second = np.polyfit(t_values, y_mm, 1)[0]
    return float(velocity_mm_per_second)