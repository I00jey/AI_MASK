import os
# 터미널에 Info 메시지 숨김 (경고부터 표시)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import cvlib as cv


# 웹캠 열기
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

sample_num = 0
captured_num = 0

while webcam.isOpened():
    status, frame = webcam.read()
    sample_num = sample_num + 1

    if not status:
        break

    # 이미지 내 얼굴 검출
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if sample_num % 8 == 0:
            captured_num = captured_num + 1
            face_in_img = frame[startY:endY, startX:endX, :]
            # 마스크 착용 이미지 수집
            # cv2.imwrite("./mask/face" + str(captured_num)+'.jpg', face_in_img)
            cv2.imwrite("./no_mask/face" + str(captured_num)+'.jpg', face_in_img)

    cv2.imshow("captured frames", frame)

    # 키보드 q를 누르면 코드 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
