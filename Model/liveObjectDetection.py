
from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")


cap.set(3, 640)
cap.set(4, 480)

model_path = "/home/michal/Repos/OpenWeekProject/Model/runs/detect/train/weights/last.pt"

# model
model = YOLO(model_path)



# object classes
classNames = ["spaghetti"]


while True:
    success, img = cap.read()

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # YOLO model may require a 3 channel image, re-convert grayscale to BGR
    processed_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    results = model(processed_img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(gray_img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (76, 211, 46)
            thickness = 1

            cv2.putText(gray_img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', gray_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()