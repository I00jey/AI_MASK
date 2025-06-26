import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path_dir_no_mask = './no_mask/'
path_dir_mask = './mask/'

file_list_no_mask = os.listdir(path_dir_no_mask)
file_list_mask = os.listdir(path_dir_mask)

file_list_no_mask_num = len(file_list_no_mask)
file_list_mask_num = len(file_list_mask)

file_total_num = file_list_mask_num + file_list_no_mask_num

# 이미지 전처리
num = 0
all_img = np.float32(np.zeros((file_total_num, 224, 224, 3)))
all_label = np.float64(np.zeros((file_total_num, 1)))

# 마스크 미착용 이미지 라벨링
for img_name in file_list_no_mask:
    img_path = path_dir_no_mask + img_name
    img = load_img(img_path, target_size=(224, 224))

    n = img_to_array(img)
    n = np.expand_dims(n, axis=0)
    n = preprocess_input(n)
    all_img[num, :, :, :] = n

    # 0 -> 마스크 미착용 라벨
    all_label[num] = 0
    num = num + 1

# 마스크 착용 이미지 라벨링
for img_name in file_list_mask:
    img_path = path_dir_mask + img_name
    img = load_img(img_path, target_size=(224, 224))

    m = img_to_array(img)
    m = np.expand_dims(m, axis=0)
    m = preprocess_input(m)
    all_img[num, :, :, :] = m

    # 1 -> 마스크 착용 라벨
    all_label[num] = 1
    num = num + 1

# 데이터 set 섞기
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

all_label = all_label[indices]
all_img = all_img[indices]

# 학습 데이터 VS 테스트 데이터 분할
num_train = int(np.round(all_label.shape[0] * 0.8))
num_test = int(np.round(all_label.shape[0] * 0.2))

train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train, :, :, :]

train_label = all_label[0:num_train]
test_label = all_label[num_train:]

# 기초 훈련 모델 생성
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print('기초 모델의 레이어 수 :', len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_img, train_label, epochs=10, batch_size=16, validation_data=(test_img, test_label))

# 학습 모델 저장
model.save('model.mask_ai')

print('학습 모델 저장 완료')


# -------------------------------------------------------------------------------------------------------
# tensorflow import 코드에서 에러가 날 경우
# 테스트 코드
# from tensorflow.keras.applications.resnet50 import ResNet50
#
# model = ResNet50(weights='imagenet')
# print("모델 로딩 성공!")
# ---------------------------------------------------------------------------------------------------------