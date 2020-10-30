import cv2
import numpy as np
import dlib
from math import sqrt
from scipy.spatial import distance
from imutils import face_utils

class Face_detector_EAR:

    def __init__(self):
        self.path = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.path)
        self.THRESHOLD = 0.25
        self.FRAME_THRESH = 3

    def detect_face(self):
        (left_s , left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (right_s , right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        video = cv2.VideoCapture(0)
        frame_cnt = 0
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.detector(gray)

            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)
                left_eye = landmarks[left_s:left_end]
                right_eye = landmarks[right_s:right_end]

                l_ear = self.eye_aspect_ratio(left_eye)
                r_ear = self.eye_aspect_ratio(right_eye)

                actual_ear = (l_ear + r_ear)/2.0
                leftEyeHull = cv2.convexHull(left_eye)
                rightEyeHull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

                if actual_ear < self.THRESHOLD:
                    frame_cnt += 1
                    if frame_cnt >= self.FRAME_THRESH:
                        cv2.putText(frame, "Dorwziness Detected !!!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                else:
                    frame_cnt = 0

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        EAR = (A + B)/(2.0 * C)
        return EAR

if __name__ == "__main__":
    fd = Face_detector_EAR()
    fd.detect_face()