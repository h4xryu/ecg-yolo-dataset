import os

version = "kitdataVersion_1_nonflipped"

image_dir = f"f:/data_set_yolov7/{version}/train/images"
label_dir = f"f:/data_set_yolov7/{version}/train/labels"


# 이미지와 라벨 이름 추출
images = {
    os.path.splitext(f)[0]
    for f in os.listdir(image_dir)
    if f.endswith((".jpg", ".png"))
}
labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")}

# 라벨 없는 이미지 찾기
missing_labels = images - labels
print(f"Images without labels: {missing_labels}")

# 이미지 없는 라벨 찾기
missing_images = labels - images
print(f"Labels without images: {missing_images}")

# 라벨 없는 이미지를 제거
for missing in missing_labels:
    image_path = os.path.join(image_dir, missing + ".jpg")  # JPG 형식
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, missing + ".png")  # PNG 형식
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removed image: {image_path}")

# 이미지 없는 라벨을 제거
for missing in missing_images:
    label_path = os.path.join(label_dir, missing + ".txt")
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Removed label: {label_path}")
