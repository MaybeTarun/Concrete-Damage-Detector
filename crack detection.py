import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os

def get_box_dimensions(boxes):
    lengths = boxes[:, 2] - boxes[:, 0]
    breadths = boxes[:, 3] - boxes[:, 1]
    dimensions = np.stack((lengths, breadths), axis=1)
    return dimensions

def main():
    cap = cv2.VideoCapture(0)
    model = YOLO('models/best.pt')

    segment = sv.MaskAnnotator()
    box = sv.BoundingBoxAnnotator()
    label = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        result = model(frame)[0]
        damage = sv.Detections.from_ultralytics(result)
        frame = segment.annotate(scene=frame, detections=damage)
        frame = box.annotate(scene=frame, detections=damage)
        frame = label.annotate(scene=frame, detections=damage)

        box_dimensions = get_box_dimensions(damage.xyxy)
        print(box_dimensions)
        for dim, boxx in zip(box_dimensions, damage.xyxy):
            x, y, _, _ = boxx
            scaled_dim = [d * 0.164 for d in dim]
            text = f"{scaled_dim[0]:.2f} x {scaled_dim[1]:.2f} cm"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = int(x)
            text_y = int(y) - 25
            cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2), (text_x + text_size[0] + 2, text_y + 2), (223,40,64), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            mask = damage.mask
            if mask is not None and np.any(mask):
                area = np.sum(mask)
                print("Area of the crack: {:.2f} cm²".format(area*0.026896))
        except Exception:
            continue

        cv2.imshow("Damage Detector", frame)

        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


#Assuming distance between object and cam as 1m, 1 cm = 0.164 pixel
