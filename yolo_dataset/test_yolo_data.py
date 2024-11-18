""" 
Make yolo v7 ECG Box data 

input -> image
process -> drag & drop
output -> boxed image & box coordinates

coordinate type ( box_mid_x, box_mid_y, width, height ) - normalized value

"""

import cv2
import os
import numpy as np
from scipy.stats import norm

# 전역 변수 설정
rectangles = []  # 사각형 좌표를 저장할 리스트
start_point = None
end_point = None
drawing = False
delete_mode = False  # 삭제 모드 플래그
image = None
img_copy = None
img_with_text = None  # 텍스트가 추가된 이미지



# WARP #################################################################################################################

def warp_image(image: np.ndarray, amplitude_x: float = 5, amplitude_y: float = 5):
    # 원본 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 사인파 주기 설정
    frequency_x = 1 / (width / 2)  # 수평 방향 주기를 이미지 폭의 한 파장으로 설정
    frequency_y = 1 / height  # 수직 방향 주기를 이미지 높이의 반파장으로 설정

    # 여유 공간을 추가하여 잘리는 부분 방지
    new_height = int(height + 2 * amplitude_y)
    new_width = int(width + 2 * amplitude_x)
    warped_image = np.zeros((new_height, new_width, *image.shape[2:]), dtype=image.dtype)

    # 이미지 중심 위치 계산
    y_offset = amplitude_y
    x_offset = amplitude_x

    # 각 픽셀 위치를 사인 함수에 따라 이동
    for y in range(height):
        for x in range(width):
            # 사인 함수를 사용하여 수평 및 수직 이동량 계산
            offset_x = int(amplitude_x * np.sin(2 * np.pi * frequency_y * y))  # 높이 기준 반파장
            offset_y = int(amplitude_y * np.sin(2 * np.pi * frequency_x * x))  # 폭 기준 한 파장

            # 새로운 좌표에 이미지 픽셀 배치 (여유 공간을 포함한 좌표)
            new_x = x + x_offset + offset_x
            new_y = y + y_offset + offset_y

            # 좌표가 유효한 범위 내에 있을 때만 픽셀 배치
            if 0 <= new_x < new_width and 0 <= new_y < new_height:
                warped_image[new_y, new_x] = image[y, x]

    return warped_image

def get_warp(image_list):
    warped_list = []
    for image in image_list:
        warped_list.append(warp_image(image))
    return warped_list

# YOLO 형식 좌표 불러오기 함수
def load_yolo_data(file_path, img_shape):
    h, w = img_shape[:2]  # 실제 이미지 크기
    boxes = []
    classes = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) != 5:
                    continue
                class_id = data[0]  # 클래스 ID
                x_center, y_center, box_width, box_height = map(float, data[1:])

                # YOLO 좌표는 정규화된 값이므로 이미지 크기에 맞게 변환
                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)
                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)

                boxes.append(((x1, y1), (x2, y2)))  # 좌상단과 우하단 좌표로 변환
                classes.append(class_id)
    return classes, boxes


# YOLO 형식으로 좌표 저장 함수
def save_yolo_data(rectangles, img_shape, output_txt):
    h, w = img_shape[:2]
    with open(output_txt, "w") as f:
        for start, end in rectangles:
            x1, y1 = start
            x2, y2 = end

            # 좌표 정렬
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            # YOLO 형식 좌표 계산
            box_mid_x = ((x_min + x_max) / 2) / w
            box_mid_y = ((y_min + y_max) / 2) / h
            box_width = (x_max - x_min) / w
            box_height = (y_max - y_min) / h

            # 파일에 저장 (클래스 ID는 1로 설정)
            f.write(
                f"1 {box_mid_x:.6f} {box_mid_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            )


# YOLO 좌표를 텍스트로 변환하는 함수
def yolo_to_text(rectangles, img_shape, classes):
    h, w = img_shape[:2]
    text_lines = []
    for i, (start, end) in enumerate(rectangles):
        x1, y1 = start
        x2, y2 = end

        # 좌표 정렬
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # YOLO 형식 좌표 계산
        box_mid_x = ((x_min + x_max) / 2) / w
        box_mid_y = ((y_min + y_max) / 2) / h
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h

        # 텍스트 라인 생성
        text_lines.append(
            f"{classes[i]} {box_mid_x:.6f} {box_mid_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return text_lines


# 텍스트를 이미지에 그리는 함수
def draw_text_on_image(img_with_text, text_lines):
    y0, dy = 30, 25  # 텍스트 시작 위치와 줄 간격
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(
            img_with_text,
            line,
            (img_with_text.shape[1] - 400, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# 이미지에 텍스트 영역을 추가하는 함수
def add_text_area_to_image(img):
    h, w, _ = img.shape
    new_w = w + 400  # 오른쪽에 400px 공간 추가
    img_with_text = np.zeros((h, new_w, 3), dtype=np.uint8)
    img_with_text[:, :w] = img  # 기존 이미지를 복사
    return img_with_text


# 마우스 이벤트 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img_copy, img_with_text, rectangles, delete_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if delete_mode:
            # 클릭한 좌표가 어떤 사각형 내부에 있는지 확인
            for i, (start, end) in enumerate(rectangles):
                x1, y1 = start
                x2, y2 = end
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    # 사각형 삭제
                    rectangles.pop(i)
                    # 이미지 업데이트
                    img_copy = image.copy()
                    img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
                    for rect in rectangles:
                        cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
                    text_lines = yolo_to_text(rectangles, image.shape)
                    draw_text_on_image(img_with_text, text_lines)  # 텍스트 갱신
                    cv2.imshow("Image", img_with_text)
                    break  # 한 개의 사각형만 삭제
        else:
            # 새로운 사각형 그리기 시작
            drawing = True
            start_point = (x, y)
            end_point = start_point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = image.copy()
            img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
            # 기존 사각형들 그리기
            for rect in rectangles:
                cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
            # 현재 그리는 사각형 그리기
            cv2.rectangle(img_with_text, start_point, end_point, (0, 0, 200), 2)
            cv2.imshow("Image", img_with_text)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_point = (x, y)
            rectangles.append((start_point, end_point))
            img_copy = image.copy()
            img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
            # 모든 사각형 다시 그리기
            for rect in rectangles:
                cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
            # YOLO 형식의 텍스트 변환 및 이미지에 출력
            text_lines = yolo_to_text(rectangles, image.shape)
            draw_text_on_image(img_with_text, text_lines)
            cv2.imshow("Image", img_with_text)


# 이미지 로드 및 박스 불러오기
def load_image_and_boxes(img_path, yolo_file):
    global image, img_copy, img_with_text, rectangles
    image = cv2.imread(img_path)
    image = warp_image(image=image, amplitude_x=5,amplitude_y=5)
    if image is None:
        print("Error: 이미지를 로드할 수 없습니다.")
        return
    img_copy = image.copy()
    img_with_text = add_text_area_to_image(img_copy)  # 텍스트 영역 추가
    classes, rectangles = load_yolo_data(yolo_file, image.shape)
    # 모든 사각형 그리기
    for i, rect in enumerate(rectangles):
        cv2.putText(img=img_with_text, text=classes[i], org=(rect[0][0],rect[0][1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0, 0, 0),thickness=3)
        cv2.rectangle(img_with_text, rect[0], rect[1], (0, 200, 0), 2)
    # YOLO 형식의 텍스트 변환 및 이미지에 출력
    text_lines = yolo_to_text(rectangles, image.shape, classes)
    draw_text_on_image(img_with_text, text_lines)


# 이미지 표시 및 키 이벤트 처리
def display_image_with_boxes(output_txt):
    global img_with_text, delete_mode
    while True:
        # img_with_text = cv2.resize(img_with_text, (960, 540))
        cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", width=1280, height=620)
        cv2.imshow("Image", img_with_text)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            save_yolo_data(rectangles, image.shape, output_txt)
            print(f"좌표가 {output_txt} 파일에 저장되었습니다.")
            break
        elif key == ord("d"):
            delete_mode = not delete_mode
            if delete_mode:
                print("삭제 모드 활성화")
            else:
                print("삭제 모드 비활성화")
        elif key == ord("p"):
            print("문제가 생긴 포인트")
            break


# if __name__ == "__main__":
#     directory = os.path.join(os.getcwd(), "camera_roi", "Version_3", "0312")
#     output_directory = os.path.join(os.getcwd(), "output", "Version_3", "0312")
#     ref_directory = os.path.join(os.getcwd(), "output", "Version_1", "0312")

#     os.makedirs(output_directory, exist_ok=True)

#     file_list = []
#     if os.path.exists(directory):
#         for file_name in os.listdir(directory):
#             file_path = os.path.join(directory, file_name)
#             if os.path.isfile(file_path):
#                 file_list.append(file_name)
#         file_list.sort()

#     ref_list = []
#     if os.path.exists(ref_directory):
#         for file_name in os.listdir(ref_directory):
#             file_path = os.path.join(ref_directory, file_name)
#             if os.path.isfile(file_path):
#                 ref_list.append(file_name)
#         ref_list.sort()

#     for file_name, reference_txt in zip(file_list, ref_list):
#         input_image_path = f"{directory}/{file_name}"
#         output_txt_path = f"{output_directory}/{file_name.split('.')[0]}.txt"
#         reference_path = f"{ref_directory}/{reference_txt}"
#         load_image_and_boxes(input_image_path, reference_path)
#         cv2.namedWindow("Image")
#         cv2.setMouseCallback("Image", draw_rectangle)
#         display_image_with_boxes(output_txt=output_txt_path)
#         cv2.destroyAllWindows()
#         print(input_image_path)
#         print(output_txt_path)
#         print(f"Reference Box : {reference_txt}")
#         pass


# 메인 실행 함수
if __name__ == "__main__":

    directory = os.path.join('e:/data_set_yolov7',"Version_1","train","image")
    output_directory = os.path.join('e:/data_set_yolov7', "output", "Version_4")
    os.makedirs(output_directory, exist_ok=True)
    file_list = []


    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_name)
        file_list.sort()


    support_txt = "E:\\data_set_yolov7\\Version_1\\train\\label"
    for file_name in file_list:
        input_image_path = f"{directory}/{file_name}"
        output_txt_path = f"{output_directory}/{file_name.split('.')[0]}.txt"
        print(os.path.join(support_txt,file_name))
        load_image_and_boxes(input_image_path, os.path.join(support_txt,f"{file_name.split('.')[0]}.txt"))
        cv2.namedWindow("Image", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", width=1280, height=740)
        cv2.setMouseCallback("Image", draw_rectangle)
        display_image_with_boxes(output_txt=output_txt_path)
        cv2.destroyAllWindows()

        print(input_image_path)
        print(output_txt_path)
        
