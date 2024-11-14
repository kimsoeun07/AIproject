import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
import numpy as np
import json
import re

def TextDetection_model(input_shape):
    inputs = Input(shape=input_shape)
    
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

def _load_image_and_label(img_file, image_folder, label_folder):
    img_path = os.path.join(image_folder, img_file)
    json_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.json')
    
        # 경로가 존재하는지 디버깅
    if os.path.exists(img_path):
        print(f"Image file exists: {json_path}")
    else:
        print(f"Image file does not exist: {json_path}")
    
    # Check if both image and JSON label files exist
    if os.path.exists(img_path) and os.path.exists(json_path):
        
        print("*********************************I")
        
        # Load and preprocess the image
        image = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        image = tf.image.resize(image, (128, 128))
        image = image / 255.0  # Normalize to [0, 1]

        # Load label data from JSON
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        text_exists = int(bool(label_data["points"]["exterior"]))  # 1 if text exists, else 0
        print('text_exists : ', text_exists)
        return image, text_exists
    
    else:
        print("Missing file:", img_file)  # 누락된 파일 출력
        return None, None  # Return None if either file is missing

def preprocess_data(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    
    # image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x.split('.')[0])[0]) if re.findall(r'\d+', x.split('.')[0]) else float('inf'))

    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    def map_fn(img_file):
        img_file_str = img_file.numpy().decode("utf-8")  # 이미지 파일 경로 디코드
        
        image, label = _load_image_and_label(img_file_str, image_folder, label_folder)  # 이미지 및 레이블 로드

        # 누락된 파일에 대한 처리
        if image is None or label is None:
            print("Missing file, skipping this item.")  # 파일 누락 시 출력
            # return tf.zeros([128, 128, 3], dtype=tf.float32), tf.constant(0, dtype=tf.int32)  # 기본 값 반환
            return None, None
        
        return image, label

    # Step 4: Apply map_fn without py_function
    # dataset = dataset.map(lambda x: tf.py_function(map_fn, [x], [tf.float32, tf.int32]),
    #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda x: tf.py_function(lambda y: map_fn(y.numpy()), [x], [tf.float32, tf.int32]))
    
    print("dataset : ", dataset)
    
    # Step 5: Filter out None values and set shape
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, [128, 128, 3]),
                                                tf.ensure_shape(label, [])))
    
    # Step 6: Set shapes and batch
    dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# 폴더 경로 설정
image_folder = 'total-text-DatasetNinja/train/img'
label_folder = 'total-text-DatasetNinja/train/ann'

# Check the dataset output
train_dataset = preprocess_data(image_folder, label_folder)

# 데이터셋의 요소를 한 번 출력하여 제대로 슬라이싱되었는지 확인
for batch in train_dataset.take(1):
    print("Batch:", batch)  # 기대하는 형식이 출력되는지 확인

# dataset_size = len(os.listdir(image_folder))
# val_size = int(0.2 * dataset_size)
# train_size = dataset_size - val_size

# train_dataset = train_dataset.shuffle(dataset_size)

# val_dataset = train_dataset.take(val_size)
# train_dataset = train_dataset.skip(val_size)

# train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
# val_dataset = val_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


# model.fit(train_dataset, epochs=10, validation_data=val_dataset)