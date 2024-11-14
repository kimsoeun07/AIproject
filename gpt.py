import os
import tensorflow as tf
import json
import re

def _load_image_and_label(img_file, image_folder, label_folder):
    img_path = os.path.join(image_folder, img_file)
    json_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.json')
    
    # 이미지 파일과 JSON 파일이 모두 존재하는지 확인
    if os.path.exists(img_path) and os.path.exists(json_path):
        # 이미지 로드 및 전처리
        image = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
        image = tf.image.resize(image, (128, 128))
        image = image / 255.0  # Normalize to [0, 1]

        # JSON 레이블 데이터 로드
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        text_exists = int(bool(label_data["points"]["exterior"]))  # 1 if text exists, else 0
        
        return image, text_exists
    else:
        # 파일이 누락된 경우
        print(f"Missing file: {img_file}")
        return None, None  # 누락된 파일은 처리하지 않음

def preprocess_data(image_folder, label_folder):
    image_files = os.listdir(image_folder)
    
    # 이미지 파일 이름에서 숫자만 추출하여 정렬
    image_files.sort(key=lambda x: int(re.findall(r'\d+', x.split('.')[0])[0]) if re.findall(r'\d+', x.split('.')[0]) else float('inf'))

    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    def map_fn(img_file):
        img_file_str = img_file.numpy().decode("utf-8")  # 이미지 파일 경로 디코드
        
        image, label = _load_image_and_label(img_file_str, image_folder, label_folder)  # 이미지 및 레이블 로드

        # 이미지 또는 레이블이 None인 경우, None을 반환하여 해당 항목을 건너뜀
        if image is None or label is None:
            return None, None  # 누락된 데이터는 None으로 반환

        return image, label

    # Step 1: map_fn을 사용하여 이미지 및 레이블 로드
    dataset = dataset.map(lambda x: tf.py_function(map_fn, [x], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Step 2: None 값을 포함하는 항목을 필터링하여 건너뜀
    dataset = dataset.filter(lambda x, y: tf.logical_and(tf.not_equal(x, None), tf.not_equal(y, None)))  # None이 아닌 데이터만 반환

    # Step 3: 배치 및 형태 설정
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, [128, 128, 3]), tf.ensure_shape(label, [])))
    dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# 폴더 경로 설정
image_folder = 'total-text-DatasetNinja/train/img'
label_folder = 'total-text-DatasetNinja/train/ann'

# 데이터셋 출력 확인
train_dataset = preprocess_data(image_folder, label_folder)

# 데이터셋의 요소를 한 번 출력하여 제대로 슬라이싱되었는지 확인
for batch in train_dataset.take(1):
    print("Batch:", batch)  # 기대하는 형식이 출력되는지 확인
