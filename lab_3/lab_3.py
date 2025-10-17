import os.path

import easyocr
from ultralytics import YOLO
import cv2

COCO_LABELS_BY_ID = {
    0: "person",
    2: "car",
    67: "cell phone"
}
OCR_READER = easyocr.Reader(["en"], verbose=False, gpu=True)

def detect_objects(model, image, labels_by_id, conf_threshold=0.5):
    results = model(image, verbose=False, half=True)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    label_ids = results.boxes.cls.cpu().numpy().astype(int)

    detections = []

    for box, conf, label_id in zip(boxes, confs, label_ids):
        if conf < conf_threshold or label_id not in labels_by_id:
            continue

        detections.append({
            "box": box.tolist(),
            "confidence": float(conf),
            "label_name": labels_by_id[label_id]
        })

    return detections


def detect_car_plates(car_plate_detector, image, conf_threshold=0.5):
    result = []
    predicted_car_plates = car_plate_detector(image, verbose=False, half=True)[0]

    for license_plate in predicted_car_plates.boxes.data.tolist():
        x1, y1, x2, y2, confidence, _ = license_plate

        if confidence < conf_threshold:
            continue

        car_plate_roi = image[int(y1):int(y2), int(x1):int(x2)]

        gray_roi = cv2.cvtColor(car_plate_roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.bilateralFilter(gray_roi, 9, 17, 17)
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = OCR_READER.readtext(thresh, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

        for (box, text, conf) in results:
            result.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "label_name": f"car plate: {text}",
            })

    return result


def draw_boxes(image, detections, box_color=(0, 255, 0), text_color=(0, 0, 0)):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        label = f"{detection['label_name']} {detection['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline), (x1 + tw, y1), box_color, -1)

        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX,0.6, text_color, 1, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    coco_detector = YOLO(os.path.join(".", "models", "yolov8n.pt"))
    car_plate_detector = YOLO(os.path.join(".", "models", "license_plate_detector.pt"))

    video_path = os.path.join(".", "input", "input_video_3.mp4")
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Cannot open video {video_path}")
        exit(1)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        objects_detections = detect_objects(coco_detector, frame, COCO_LABELS_BY_ID)
        car_plates_detections = detect_car_plates(car_plate_detector, frame)

        detections_to_show = objects_detections + car_plates_detections

        frame_to_show = draw_boxes(frame, detections_to_show)

        cv2.imshow("Video", frame)
        cv2.waitKey(0)