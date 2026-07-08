import cv2

cap = cv2.VideoCapture(0)

print("opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    print("ret:", ret, "frame is None:", frame is None)

    if frame is not None:
        cv2.imshow("cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()