import cv2


def video_predict(facedetector_fn, landmark_model):
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        rects = facedetector_fn(img)

        for rect in rects:
            marks = detect_marks(img, landmark_model, rect)
            draw_marks(img, marks)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

