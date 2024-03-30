import argparse
import supervision as sv
import cv2
import numpy as np
from inference import get_model
from collections import defaultdict, deque

# Jika tidak ada mobil lewat

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(
                                        description="Viachle speed estimation using inference and supervision"
                                    )
    parser.add_argument(
        "-p","--source_video_path",
        required=True,
        help="path to the source video",
        type=str
    )
    return parser.parse_args()
    

def main():
    args = parse_args()
    model = get_model(model_id="yolov8n-640")
    
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    
    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)//3
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)/6
    print(text_scale)
    print(video_info.fps)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_anotator = sv.LabelAnnotator(text_scale=text_scale, 
                                       text_thickness=thickness,
                                       color_lookup=sv.ColorLookup.TRACK)
    trace_anotator = sv.TraceAnnotator(thickness=thickness, 
                                       trace_length=video_info.fps*2, 
                                       position=sv.Position.BOTTOM_CENTER,
                                       color_lookup=sv.ColorLookup.TRACK)
    
    POLOGON_SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])//5
    
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 250
    TARGET = np.array([[0, 0], 
                       [TARGET_WIDTH - 1, 0], 
                       [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], 
                       [0, TARGET_HEIGHT - 1]])
    
    class ViewTransformer:
        def __init__(self, source: np.ndarray, target:np.ndarray) -> None:
            source = source.astype(np.float32)
            target = target.astype(np.float32)
            self.m = cv2.getPerspectiveTransform(source, target)
        
        def transform_point(self, point: np.ndarray) -> np.ndarray:
            reshape_points = point.reshape(-1, 1, 2).astype(np.float32)
            transform_points = cv2.perspectiveTransform(reshape_points, self.m)
            return transform_points.reshape(-1, 2) 
    
    polygon_zone = sv.PolygonZone(polygon=POLOGON_SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_trainsformer = ViewTransformer(source=POLOGON_SOURCE, target=TARGET)
    
    coordinate = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    for frame in frame_generator:
        height, widht = frame.shape[:2]
        frame = cv2.resize(frame, (widht//5, height//5))
        result = model.infer(frame)[0]
        detector = sv.Detections.from_inference(result)
        detector = detector[polygon_zone.trigger(detector)]
        detector = byte_track.update_with_detections(detections=detector)
        points = detector.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        if len(points) != 0:
            points = view_trainsformer.transform_point(point=points)
            
            labels = []
            for tracker_id, [_, y] in zip(detector.tracker_id, points):
                coordinate[tracker_id].append(y)
                
                if len(coordinate[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinate[tracker_id][-1]
                    coordinate_end = coordinate[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinate[tracker_id])/video_info.fps
                    speed = distance/time*3.6
                    labels.append(f"#{tracker_id}: {speed:.2f} km/h") 
            
            annatoted_frame = frame.copy()
            # annatoted_frame = sv.draw_polygon(annatoted_frame, 
            #                                 polygon=POLOGON_SOURCE, 
            #                                 color=sv.Color.RED)
            annatoted_frame = trace_anotator.annotate(scene=annatoted_frame, 
                                                    detections=detector)
            annatoted_frame = bounding_box_annotator.annotate(scene=annatoted_frame, 
                                                            detections=detector)
            annatoted_frame = label_anotator.annotate(scene=annatoted_frame, 
                                                    detections=detector, 
                                                    labels=labels)
            
        cv2.imshow("annatoted frame", annatoted_frame)
        if cv2.waitKey(1) == 27:
            break
        
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
        
        
    
    
    