from utils.timers import FPSBasedTimer
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

VIDEO_PATH = r"data\demo.mp4"
MODEL_PATH = r"model\best.pt"

model = YOLO(MODEL_PATH)

tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
fps_monitor = sv.FPSMonitor()
# timers = [FPSBasedTimer(video_info.fps) for _ in zones]

frame_generator = sv.get_video_frames_generator(source_path=VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
WIDTH = 1020
HEIGHT = 500

ZONE_POLYGON = [[[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]]
ZONE_POLYGON = [np.array(polygon, np.int32) for polygon in ZONE_POLYGON]

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000"))

for frame in frame_generator:
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    results = model(
        frame, verbose=False, conf=0.5, iou=0.7
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    zones = [
        sv.PolygonZone(
            polygon=polygon,
            # frame_resolution_wh=(WIDTH, HEIGHT),
            triggering_anchors=(sv.Position.CENTER,)
        )
        for polygon in ZONE_POLYGON
    ]

    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    fps_monitor.tick()
    fps = fps_monitor.fps

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )

    # for idx, zone in enumerate(zones):
    # annotated_frame = sv.draw_polygon(
    #     scene=annotated_frame,
    #     polygon=zone.polygon,
    #     color=COLORS.by_idx(idx)
    # )

    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text=f"fps: {fps:.1f}",
        text_anchor=sv.Point(WIDTH - 40, 30),
        background_color=sv.Color.from_hex("#A351FB"),
        text_color=sv.Color.from_hex("#000000"),
    )

    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    cv2.imshow("Processed Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
