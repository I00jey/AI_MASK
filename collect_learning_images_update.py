import os
# 경고 메시지만 출력
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp


# 저장할 폴더 설정 (mask와 no_mask
SAVE_DIR = './no_mask'
# SAVE_DIR = './mask'

# 디렉토리 없으면 생성
os.makedirs(SAVE_DIR, exist_ok=True)

# 웹캠 열기
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print('카메라를 열 수 없습니다.')
    exit()

# mediapipe 얼굴 탐지 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

sample_num = 9
captured_num = 0

while True:
    status, frame = webcam.read()
    if not status:
        print("프레임을 읽을 수 없습니다.")
        break

    sample_num += 1

    # BGR -> RGB 변환 (mediapipe는 RGB 입력 요구)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    detections = getattr(results, 'detections', None)
    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            x2 = int((bbox.xmin + bbox.width) * iw)
            y2 = int((bbox.ymin + bbox.height) * ih)

            # 프레임 간격에 따라 이미지 저장
            if sample_num % 8 == 0:
                captured_num += 1
                face_img = frame[y1:y2, x1:x2]

                filename = os.path.join(SAVE_DIR, f'face{captured_num}.jpg')
                cv2.imwrite(filename, face_img)
                print(f"[{captured_num}] 저장됨: {filename}")

            # 사각형 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow("collecting_learning_images", frame)

    # 종료: 'q' 또는 'ESC' 키
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------
# 프로젝트 코드 업데이트 요약
#  Mediapipe 얼굴 탐지	cvlib보다 빠르고 정확하며, 유지보수도 활발
#  자동 폴더 생성	os.makedirs(..., exist_ok=True)
#  샘플 저장 주기	8프레임마다 저장 (sample_num % 8 == 0)
#  예외 방지	getattr(results, 'detections', None)로 안전한 속성 접근
#  UI 개선	실시간 프레임에 사각형 표시, 저장 로그 출력