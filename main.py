# main.py
import cv2
from pathlib import Path

from detector import detect_drops
from tracker import SingleDropTracker
from velocity import calculate_velocity_with_grid
from grid_mask import build_grid_mask, is_near_grid
from grid_calibration import detect_horizontal_grid_lines, draw_detected_grid_lines
from config import MAX_TRAJECTORY


VIDEO_DIR = Path("data/videos")
GRID_DISTANCE_MM = 0.25
SHOW_GRID_DEBUG = False


def select_initial_point(frame, video_name):
    """
    鼠标点击选择目标油滴。

    操作方式：
    左键点击目标油滴
    Enter 或 Space：确认
    s：跳过当前视频
    ESC：退出程序
    """
    selected = []
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN:
            selected.clear()
            selected.append((x, y))

            display = frame.copy()
            cv2.circle(display, (x, y), 12, (0, 255, 0), 2)
            cv2.circle(display, (x, y), 2, (0, 0, 255), 3)

    window_name = f"Select target - {video_name}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display.copy()

        cv2.putText(
            temp,
            "Click target drop, Enter/Space=confirm, s=skip, ESC=exit",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10, 32]:
            cv2.destroyWindow(window_name)
            if len(selected) > 0:
                return selected[0]
            return None

        if key in [ord("s"), ord("S")]:
            cv2.destroyWindow(window_name)
            return None

        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit


def get_video_files(video_dir):
    """
    读取文件夹中的所有视频文件。
    """
    video_files = []
    video_files += list(video_dir.glob("*.mp4"))
    video_files += list(video_dir.glob("*.MP4"))
    video_files += list(video_dir.glob("*.mov"))
    video_files += list(video_dir.glob("*.MOV"))

    video_files = sorted(set(video_files))
    return video_files


def main():
    video_files = get_video_files(VIDEO_DIR)

    print(f"一共找到 {len(video_files)} 个视频")

    if len(video_files) == 0:
        print("没有找到视频，请检查 data/videos 文件夹。")
        return

    cv2.namedWindow("Millikan AI System", cv2.WINDOW_NORMAL)

    for video_path in video_files:
        print("\n==============================")
        print(f"正在处理视频：{video_path}")
        print("==============================")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"视频打开失败：{video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 60.0

        print("FPS:", fps)

        ret, first_frame = cap.read()
        if not ret:
            print("无法读取第一帧，跳过该视频。")
            cap.release()
            continue

        grid_y_pixels = detect_horizontal_grid_lines(first_frame)

        print("自动检测到横向网格线数量：", len(grid_y_pixels))
        print("横向网格线 y 坐标：", grid_y_pixels)

        if len(grid_y_pixels) < 3:
            print("横向网格线检测失败，跳过该视频。")
            cap.release()
            continue

        if SHOW_GRID_DEBUG:
            debug_frame = draw_detected_grid_lines(first_frame, grid_y_pixels)
            cv2.imshow("Detected Grid Lines", debug_frame)
            cv2.waitKey(800)

        grid_mask = build_grid_mask(first_frame)

        init_point = select_initial_point(first_frame, video_path.name)
        if init_point is None:
            print("跳过该视频。")
            cap.release()
            continue

        tracker = SingleDropTracker(
            max_trajectory=MAX_TRAJECTORY,
            max_jump=35,
            init_point=init_point,
            max_miss=12,
            base_radius=80,
            max_radius=140
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()
            h, w = frame.shape[:2]

            search_center = tracker.get_search_center()
            search_radius = tracker.get_search_radius()

            if search_center is not None:
                sx, sy = search_center
                if sx < 0 or sx >= w or sy < 0 or sy >= h:
                    search_center = tracker.last_point

            near_grid = is_near_grid(
                grid_mask,
                search_center,
                radius=14
            )

            if near_grid:
                target, trajectory = tracker.predict_only()
            else:
                drops = detect_drops(
                    frame,
                    search_center=search_center,
                    search_radius=search_radius
                )

                drops = [
                    d for d in drops
                    if not is_near_grid(grid_mask, (d[0], d[1]), radius=8)
                ]

                target, trajectory = tracker.update(drops)

            if tracker.last_point is not None:
                lx, ly = tracker.last_point

                if 0 <= lx < w and 0 <= ly < h:
                    cv2.circle(
                        frame,
                        tracker.last_point,
                        search_radius,
                        (255, 255, 0),
                        1
                    )

            if target is not None:
                x, y, r = target
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            for i in range(1, len(trajectory)):
                x1, y1, m1 = trajectory[i - 1]
                x2, y2, m2 = trajectory[i]

                color = (255, 0, 0) if (m1 and m2) else (100, 180, 255)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

            if len(trajectory) >= 15:
                velocity = calculate_velocity_with_grid(
                    trajectory,
                    fps,
                    grid_y_pixels,
                    grid_distance_mm=GRID_DISTANCE_MM
                )
            else:
                velocity = 0.0

            cv2.putText(
                frame,
                f"Velocity: {velocity:.4f} mm/s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Missed: {tracker.missed}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

            if tracker.missed > 8:
                cv2.putText(
                    frame,
                    "Tracking lost, press R to reselect",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Millikan AI System", frame)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

            if key in [ord("r"), ord("R")]:
                print("重新选择油滴...")

                init_point = select_initial_point(raw_frame, video_path.name)

                if init_point is not None:
                    tracker = SingleDropTracker(
                        max_trajectory=MAX_TRAJECTORY,
                        max_jump=55,
                        init_point=init_point,
                        max_miss=12,
                        base_radius=80,
                        max_radius=140
                    )
                    print("重新选择完成：", init_point)
                else:
                    print("没有重新选择，继续当前视频。")

        cap.release()

    cv2.destroyAllWindows()
    print("全部视频处理完成。")


if __name__ == "__main__":
    main()