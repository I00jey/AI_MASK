import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

# 학습모델 불러오기
model = load_model('모델명')
model.summary()

# 카메라 열기
webcam = cv2.VideoCapture(0)

# 카메라 오류 처리
if not webcam.isOpened():
    print('Could not open camera')
    exit()

# 카메라 실행 성공시
while webcam.isOpened():

    status, frame = webcam.read()
    if not status:
        print("could not read frame")
        exit()

    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region1 = frame[startY:endY, startX:endX]
            face_region2 = cv2.resize(face_region1, (224, 224), interpolation=cv2.INTER_AREA)

            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)

            # 마스크 미착용 case
            if prediction < 0.7:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 225), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "마스크를 착용하지 않았습니다 {:.1f}%".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 마스크 착용 case
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "마스크를 착용하셨습니다 {:.1f}%".format((prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 디스플레이 출력
    cv2.imshow("wearing mask determination camera", frame)

    # 카메라 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
