import cv2
import dlib

# Загрузка предобученной модели для обнаружения лиц
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Функция для обнаружения и отслеживания глаз
def get_eye_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y
        return left_eye, right_eye
    return None, None

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    left_eye, right_eye = get_eye_region(frame)
    if left_eye and right_eye:
        cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
