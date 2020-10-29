import cv2
import numpy as np
import dlib
from math import sqrt
from keras.models import load_model

class Face_Detector():

    def __init__(self):
        # self.file = file
        self.model = load_model("best_model (1).h5")

    def detect_face(self):
        video = cv2.VideoCapture(0)
        landmark_file = "shape_predictor_68_face_landmarks.dat"
        # classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor(landmark_file)

        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

            ############# when using Cascade ###################
            # faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)

            faces = face_detector(gray)

            ############ when using Cascade #####################
            # for (x,y,w,h) in faces:

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                ############### when using Cascade ###################
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

                ####### when using Cascade #############
                # roi = gray[y:y+h, x:x+w]
                # roi_color = frame[y:y+h, x:x+w]

                ######### when using dlib ##############
                roi = gray[y1:y2, x1:x2]
                roi_color = frame[y1:y2, x1:x2]

                landmarks = landmark_predictor(gray, face)

                # left = (landmarks[36], landmarks[39])
                left_dist = sqrt((landmarks.part(39).x-landmarks.part(36).x)**2 + (landmarks.part(39).y-landmarks.part(36).y)**2)
                # right = (landmarks[42], landmarks[45])
                right_dist = sqrt((landmarks.part(45).x-landmarks.part(42).x)**2 + (landmarks.part(45).y-landmarks.part(42).y)**2)
                left_eye_roi = frame[int(landmarks.part(36).y-(left_dist/1.5)):int(landmarks.part(39).y+(left_dist/1.5)) , int(landmarks.part(36).x-(left_dist/2)):int(landmarks.part(39).x+(left_dist/2))]
                left_eye_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
                right_eye_roi = frame[int(landmarks.part(42).y-(right_dist/1.5)):int(landmarks.part(45).y+(right_dist/1.5)) ,int(landmarks.part(42).x-(right_dist/2)):int(landmarks.part(45).x+(right_dist/2))]
                right_eye_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
                
                cv2.rectangle(frame,
                              (int(landmarks.part(36).x-(left_dist/1.5)),int(landmarks.part(36).y-(left_dist/1.5))),
                              (int(landmarks.part(39).x+(left_dist/1.5)),int(landmarks.part(39).y+(left_dist/1.5))),
                              (0, 255, 0), 3)
                cv2.rectangle(frame,
                              (int(landmarks.part(42).x - (right_dist / 1.5)), int(landmarks.part(42).y - (right_dist/1.5 ))),
                              (int(landmarks.part(45).x + (right_dist / 1.5)), int(landmarks.part(45).y + (right_dist/1.5 ))),
                              (0, 255, 0), 3)

                left_eye_roi[: ,: ,0] = left_eye_gray
                left_eye_roi[:, :, 1] = left_eye_gray
                left_eye_roi[:, :, 2] = left_eye_gray

                right_eye_roi[:, :, 0] = right_eye_gray
                right_eye_roi[:, :, 1] = right_eye_gray
                right_eye_roi[:, :, 2] = right_eye_gray

                left_eye_img = cv2.resize(left_eye_roi, (64, 64))
                # left_eye_img = np.atleast_3d(left_eye_img)
                left_eye_img = np.expand_dims(left_eye_img, axis=0)

                right_eye_img = cv2.resize(right_eye_roi, (64, 64))
                # right_eye_img = np.atleast_3d(right_eye_img)
                right_eye_img = np.expand_dims(right_eye_img, axis=0)

                left_eye_img = cv2.normalize(left_eye_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                right_eye_img = cv2.normalize(right_eye_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                left_eye_score = self.model.predict(left_eye_img)
                right_eye_score = self.model.predict(right_eye_img)
                print((left_eye_score, right_eye_score))

                if left_eye_score[0] < 0.3 and right_eye_score[0] < 0.3:
                    cv2.putText(frame, "Dorwziness Detected !!!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # cv2.circle(frame, (x,y), 1, (0, 0, 255), 1)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    det = Face_Detector()
    det.detect_face()