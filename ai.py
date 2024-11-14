import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
import numpy as np
import json
import re

def TextDetection_model(input_shape):
    inputs = Input(shape = input_shape)
    
    # Convolutional Layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # 입력을 첫 레이어로 전달
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Text 여부를 이진 분류

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = TextDetection_model((128, 128, 3))
# model.summary()

# 데이터 전처리 함수 (배치 처리로 수정)
def preprocess_data(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    
    # 숫자 기준으로 정렬 (파일명에서 숫자 부분만 추출하여 정렬)
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 파일명에서 숫자만 추출하여 정렬
    

    def _load_image_and_label(img_file):
        img_path = os.path.join(image_folder, img_file)
        json_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.json')

        if os.path.exists(json_path):  # JSON 라벨 파일이 있는 경우
            # 이미지 로드 및 전처리
            image = tf.image.decode_image(tf.io.read_file(img_path))
            image = tf.image.resize(image, (128, 128))
            image = image / 255.0

            # JSON 파일 로드 및 라벨링 정보 추출
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            text_exists = int(bool(label_data["points"]["exterior"]))  # 텍스트 영역 여부
            return image, text_exists

    # tf.data.Dataset을 사용하여 배치 처리
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(lambda x: tf.py_function(_load_image_and_label, [x], [tf.float32, tf.int32]))
    dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# 폴더 경로 설정
image_folder = 'total-text-DatasetNinja/train/img'
label_folder = 'total-text-DatasetNinja/train/ann'


# # 데이터 전처리 및 모델 학습
# train_images, train_labels = preprocess_data(image_folder, label_folder)
# model.fit(train_images, train_labels, epochs=10, batch_size=16, validation_split=0.2)

# 데이터 전처리 및 모델 학습
train_dataset = preprocess_data(image_folder, label_folder)
# print('train_dataset : {}'.format(train_dataset))

# Assume train_dataset is already defined from preprocess_data function
dataset_size = len(os.listdir(image_folder))  # or len(list(train_dataset)) after creating train_dataset
val_size = int(0.2 * dataset_size)  # Reserve 20% for validation
train_size = dataset_size - val_size

# Shuffle dataset before splitting
train_dataset = train_dataset.shuffle(dataset_size)

# Split into train and validation
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# Batch datasets
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


# Now use these datasets for training
# model.fit(train_dataset, epochs=10, validation_data=val_dataset)