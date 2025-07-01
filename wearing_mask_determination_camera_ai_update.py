import cv2
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
# cvlib는 유지보수가 거의 되지 않아 mediapipe로 대체
import mediapipe as mp
import time

# 마스크 분류 모델 불러오기
model = load_model('모델명.h5')
model.summary()

# mediapipe 얼굴 탐지 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 웹캠 열기
webcam = cv2.VideoCapture(0)

# 웹캠 연결 확인
if not webcam.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 실시간 영상 처리 루프
while True:
    ret, frame = webcam.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # mediapipe는 RGB 이미지 사용
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    # detections 속성이 없을 때 안전하게 무시 가능
    detections = getattr(results, 'detections', None)

    # 얼굴이 감지된 경우
    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            x2 = int((bbox.xmin + bbox.width) * iw)
            y2 = int((bbox.ymin + bbox.height) * ih)

            # 얼굴 영역 잘라내기
            face_region = frame[y1:y2, x1:x2]

            try:
                # 얼굴 영역 리사이즈 및 전처리
                face_resized = cv2.resize(face_region, (224, 224))
                x = img_to_array(face_resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # 마스크 착용 여부 예측
                prediction = model.predict(x, verbose=0)

                confidence = float(prediction[0][0])
                if confidence < 0.7:
                    # 마스크 미착용
                    color = (0, 0, 255)
                    label = f'마스크 미착용 : {100 - confidence * 100:.1f}%'

                else:
                    # 마스크 착용
                    color = (0,255,0)
                    label = f'마스크 착용 : {confidence*100:.1f}%'

                # 얼굴 박스와 텍스트 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f'처리 중 오류 발생 : {e}')
                continue

    # 화면에 출력
    cv2.imshow("마스크 착용 여부 판별 카메라", frame)

    # 'q' 키를 누르면 종료 (종료조건)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# 리소스 정리
webcam.release()
cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------
# 프로젝트 코드 업데이트 요약
# 얼굴 탐지	cvlib.detect_face() ->  mediapipe.FaceDetection
# 모델 로딩	tensorflow.keras.models  ->  keras.models (TF 2.12 이상 권장)
# 이미지 전처리	그대로 사용 유지
# 종료 조건에 Esc 키 추가