import os
import shutil
from sklearn.model_selection import train_test_split

version = "kitdataVersion_1_flipped"

# 데이터 경로 설정
label_folder = os.path.join(
    "f:", "data_set_yolov7", version, "train", "labels"
)  # 라벨 파일 폴더
image_folder = os.path.join(
    "f:", "data_set_yolov7", version, "train", "images"
)  # 이미지 파일 폴더
output_valid_folder = os.path.join("f:", "data_set_yolov7", version, "valid")
output_test_folder = os.path.join("f:", "data_set_yolov7", version, "test")
output_train_folder = os.path.join("f:", "data_set_yolov7", version, "train_split")

# Validation-Test 분할 비율
test_ratio = 0.2  # Test set 비율 (20%)
valid_ratio = 0.2  # Validation set 비율 (20% of remaining data)

# 1. 라벨 및 이미지 파일 목록 불러오기
label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]  # 라벨 파일
image_files = [
    f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))
]  # 이미지 파일

# 2. 라벨과 이미지 매칭 확인
matched_files = []
for label_file in label_files:
    image_name = os.path.splitext(label_file)[0]  # 라벨 파일에서 확장자 제거
    for image_file in image_files:
        if image_name == os.path.splitext(image_file)[0]:
            matched_files.append((label_file, image_file))
            break

# 3. Test 세트 분리
remaining_set, test_set = train_test_split(
    matched_files, test_size=test_ratio, random_state=42
)

# 4. Validation 세트 분리
train_set, valid_set = train_test_split(
    remaining_set, test_size=valid_ratio, random_state=42
)

# 5. 출력 폴더 생성
os.makedirs(output_valid_folder, exist_ok=True)
os.makedirs(os.path.join(output_valid_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_valid_folder, "images"), exist_ok=True)

os.makedirs(output_test_folder, exist_ok=True)
os.makedirs(os.path.join(output_test_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_test_folder, "images"), exist_ok=True)

os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(os.path.join(output_train_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_train_folder, "images"), exist_ok=True)

# 6. Validation 데이터 복사
for label_file, image_file in valid_set:
    shutil.copy(
        os.path.join(label_folder, label_file),
        os.path.join(output_valid_folder, "labels", label_file),
    )
    shutil.copy(
        os.path.join(image_folder, image_file),
        os.path.join(output_valid_folder, "images", image_file),
    )

# 7. Test 데이터 복사
for label_file, image_file in test_set:
    shutil.copy(
        os.path.join(label_folder, label_file),
        os.path.join(output_test_folder, "labels", label_file),
    )
    shutil.copy(
        os.path.join(image_folder, image_file),
        os.path.join(output_test_folder, "images", image_file),
    )

for label_file, image_file in train_set:
    shutil.copy(
        os.path.join(label_folder, label_file),
        os.path.join(output_train_folder, "labels", label_file),
    )
    shutil.copy(
        os.path.join(image_folder, image_file),
        os.path.join(output_train_folder, "images", image_file),
    )

print("Validation and test sets are created successfully!")
