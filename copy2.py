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
def _load_image_and_label(img_file):
    img_path = os.path.join(image_folder, img_file)
    json_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.json')
    
    print("Checking paths: *********************************************")
    print("Image path:", img_path)
    print("JSON path:", json_path)

    # Check if both image and JSON label files exist
    if os.path.exists(img_path) and os.path.exists(json_path):
        
        # Load and preprocess the image
        image = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        image = tf.image.resize(image, (128, 128))
        image = image / 255.0  # Normalize to [0, 1]

        # Load label data from JSON
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        text_exists = int(bool(label_data["points"]["exterior"]))  # 1 if text exists, else 0
        return image, text_exists
    else:
        return None, None  # Return None if either file is missing

def preprocess_data(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    
        # 데이터셋의 요소를 한 번 출력하여 제대로 슬라이싱되었는지 확인
    # for item in dataset.take(5):
    #     print("Dataset item:", item)  # 기대하는 형식이 출력되는지 확인 => 정상 출력 됨!
    
    def map_fn(img_file):
        # Decode filename to string and load image and label
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")   
        print("img_file  : {}".format(img_file))
        
        image, label = _load_image_and_label(img_file.numpy().decode("utf-8"))
        
        if image is None or label is None:
            # Return placeholder values if data is missing
            return tf.zeros([128, 128, 3], dtype=tf.float32), tf.constant(0, dtype=tf.int32)
        return image, label

    # Use tf.py_function and set output shapes explicitly
    # dataset = dataset.map(lambda x: tf.py_function(map_fn, [x], [tf.float32, tf.int32]),
    #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda x: tf.py_function(lambda y: map_fn(y.numpy()), [x], [tf.float32, tf.int32]))

    # Set shapes after mapping
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, [128, 128, 3]),
                                                tf.ensure_shape(label, [])))

    # Batch dataset only once after setting correct shapes
    dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset




# 폴더 경로 설정
image_folder = 'total-text-DatasetNinja/train/img'
label_folder = 'total-text-DatasetNinja/train/ann'

# 데이터 전처리 및 모델 학습
train_dataset = preprocess_data(image_folder, label_folder)

# print("train_dataset : {}".format(train_dataset))

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