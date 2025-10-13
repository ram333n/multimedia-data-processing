import os.path

from ultralytics import YOLO
import cv2

COCO_LABELS_BY_ID = {
    2: "car"
}

def detect_objects(model, image, labels_by_id, conf_threshold=0.5):
    results = model(image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    detections = []

    for box, conf, class_id in zip(boxes, confs, class_ids):
        if conf < conf_threshold or class_id not in labels_by_id:
            continue

        detections.append({
            "box": box.tolist(),
            "confidence": float(conf),
            "class_id": int(class_id),
            "class_name": labels_by_id[class_id]
        })

    return detections


def draw_boxes(image, detections, box_color=(0, 255, 0), text_color=(0, 0, 0)):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        label = f"{detection['class_name']} {detection['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline), (x1 + tw, y1), box_color, -1)

        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX,0.6, text_color, 1, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    video_path = os.path.join(".", "input", "input_video.mp4")
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Cannot open video {video_path}")
        exit(1)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        objects_detections = detect_objects(model, frame, COCO_LABELS_BY_ID)
        frame_to_show = draw_boxes(frame, objects_detections)

        cv2.imshow("Video", frame)
        cv2.waitKey(0)