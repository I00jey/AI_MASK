import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# 불편요한 TF 로그 숨김 (에러 이상만 표시)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 데이터 경로
path_dir_no_mask = './no_mask'
path_dir_mask = './mask'

file_list_no_mask = os.listdir(path_dir_no_mask)
file_list_mask = os.listdir(path_dir_mask)

# 총 이미지 수
file_total_num = len(file_list_no_mask) + len(file_list_mask)

# 이미지 및 라벨 초기화
all_img = np.zeros((file_total_num, 224, 224, 3), dtype=np.float32)
all_label = np.zeros((file_total_num, 1), dtype=np.float32)


# 이미지 로딩 함수
def load_and_preprocess(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


# 마스크 미착용 (label = 0)
num = 0
for img_name in file_list_no_mask:
    img_path = os.path.join(path_dir_no_mask, img_name)
    all_img[num] = load_and_preprocess(img_path)
    all_label[num] = 0
    num += 1

# 마스크 미 착용 (label = 1)
for img_name in file_list_mask:
    img_path = os.path.join(path_dir_mask, img_name)
    all_img[num] = load_and_preprocess(img_path)
    all_label[num] = 1
    num += 1

# 데이터 셋 섞고 분리
train_img, test_img, train_label, test_label = train_test_split(all_img, all_label, test_size=0.2, shuffle=True,
                                                                random_state=42)

# 모델 정의 (ResNet50 + Custom Layer)
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(
    input_shape=IMG_SHAPE,
    weights='imagenet',
    include_top=False
)

# 전이 학습 (기초 모델 고정)
base_model.trainable = False

# 전체 모델 구성
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 모델 학습
model.fit(
    train_img, train_label,
    epochs=10,
    batch_size=16,
    validation_data=(test_img, test_label)
)

# 모델 저장 (최신 포맷은 .keras 확장자)
model.save('model.mask_ai.keras')
print('✅ 학습 모델 저장 완료!')

#-------------------------------------------------------------------
# 프로젝트 코드 업데이트 요약
#  model.save('...')	.keras 확장자가 공식 권장 형식
#  train_test_split 사용	numpy 인덱싱보다 깔끔하고 안정적
#  np.zeros(..., dtype=...)	np.float64 → np.float32로 성능 최적화
#  Adam(lr=...) → learning_rate	최신 문법 반영
#  전체 구조 간소화	함수화로 가독성 향상