import argparse
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from reference.utils.general import find_in_list, load_zones_config
from reference.utils.timers import FPSBasedTimer
import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

my_file = open("classes.txt", "r")
data = my_file.read()
CLASS = data.split("\n")


def main() -> None:
    model = YOLO('best.pt')
    byte_tracker = sv.ByteTrack()
    annotator = sv.BoxAnnotator()
    frames_generator = sv.get_video_frames_generator('demo.mp4')

    for frame in frames_generator:
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame,
                                verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = byte_tracker.update_with_detections(detections)
        annotated_frame = frame.copy()
        annotated_frame = annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
