import cv2
import numpy as np
import time

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
path = 'video2.mp4'
cap = cv2.VideoCapture(path)

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    start_time = time.time()  # Start time for FPS calculation

    success, img = cap.read()
    img = cv2.resize(img, (700, 450))
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            class_id = int(classIds[i][0]) if isinstance(classIds[i], np.ndarray) else int(classIds[i])
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            
            # Get confidence score and convert to percentage
            confidence = confs[i] * 100
            label = f"{classNames[class_id - 1].upper()} {confidence:.2f}%"
            
            # Display object name and confidence score
            cv2.putText(
                img,
                label,
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"

    # Display FPS on the image
    cv2.putText(
        img,
        fps_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Show output with bounding boxes, class names, and confidence
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
