import torch
from src import plot_results, plot_results_with_target
from src import Detector, Denoising, Digitization, Classifier, ClassifierRaw
import numpy as np
import time
import os
import re
import cv2
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'로 변경 가능
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

# DEVICE = torch.device('cpu') # using CPU
DEVICE = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
start = time.time()

def sort_key(filename):
    # 파일명에서 숫자 추출 (예: 'MR000001-0_thr_12.jpg'에서 12를 추출)
    numbers = re.findall(r'\d+', filename)
    # 정렬을 위해 추출된 숫자를 정수로 변환
    return [int(num) for num in numbers]

def get_image(lead_path, path):
    image_list = []
    for file in path:
        file_path = lead_path + f'/{file}'
        image_list.append(cv2.imread(file_path,0))
    return image_list

def create_directory_if_empty(directory_path):
    # 디렉토리가 존재하지 않거나, 비어 있으면 생성
    if not os.path.exists(directory_path) or not os.listdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def get_all_directories(directory_path):
    # 주어진 디렉토리 내에서 하위 디렉토리만 가져옴
    return [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

def save_data(diagnosis_classes, data_list, extracted_result_list, target_result_list, dir, filename):
    # extracted_result_list와 target_result_list를 백분율로 변환 후 정수로 반올림
    percent_extracted = [np.round(result * 100).astype(int) for result in extracted_result_list]
    percent_target = [np.round(result * 100).astype(int) for result in target_result_list]

    # 데이터 리스트 길이 확인 및 열 이름 생성
    assert len(data_list) == len(percent_extracted) == len(percent_target), "데이터 리스트와 결과 리스트의 길이가 일치해야 합니다."

    # 각 데이터에 대해 extracted 및 target 결과를 열로 병합
    combined_results = []
    for extracted, target in zip(percent_extracted, percent_target):
        combined_results.append(np.column_stack((extracted, target)))

    # 모든 결과를 진단 클래스를 기준으로 행 방향으로 정렬
    all_results = np.column_stack(combined_results)  # 진단 클래스별로 모든 데이터 병합

    # 멀티 인덱스 설정: 첫 번째 행에 데이터 이름, 두 번째 행에 extracted/target
    columns = pd.MultiIndex.from_product([data_list, ['extracted', 'target']], names=['Data', 'Type'])

    # DataFrame 생성
    df = pd.DataFrame(all_results, index=diagnosis_classes, columns=columns)

    # 엑셀 파일로 저장
    save_dir = dir + filename
    df.to_excel(save_dir)

def draw_extracted_on_paper(extracted, papers, save_dir):
    for i in range(13):
        plt.figure()
        plt.imshow(papers[i], cmap='gray')
        plt.plot(extracted[i])
        plt.savefig(save_dir+f'image_{i}')

def add_salt_pepper_noise(image, salt_prob):
    # Create a copy of the image to avoid modifying the original
    noisy_image = image.copy()

    # Generate random noise based on the salt and pepper probabilities
    salt_noise = np.random.rand(*image.shape[:2]) < salt_prob

    # Apply salt noise (white pixels)
    noisy_image[salt_noise] = 255

    return noisy_image

def get_noisy_image(image_list):
    noise_list = []
    for image in image_list:
        noise_list.append(add_salt_pepper_noise(image,salt_prob=0.01))
    return noise_list

# MAIN #################################################################################################################
if __name__ == '__main__':
    class_num = 34  # 34가지 심전도 진단
    img_h = 1500  # 이미지 높이
    img_w = 3360  # 이미지 너비
    lead_len = 1024  # 입력 신호 길이
    lead_2_len = 4096  # Lead II 신호 길이
    diagnosis_classes = np.load('./examples/target_labels.npy', allow_pickle=True)  # 34가지 진단 레이블 및 순서

    lead_dir = './examples/examples_for_check_digitization/image_dir'
    # directories = sorted(get_all_directories(lead_dir), key=sort_key)
    directories = ['3']

    data_dir = './examples/examples_for_check_digitization/'
    extracted_results = []
    target_results = []

    for i, dir_name in enumerate(directories):
        print(f'Processing : {i}/{len(directories)}...')
        lead_path = f'./examples/examples_for_check_digitization/image_dir/{dir_name}'
        file_names = sorted([f for f in os.listdir(lead_path) if os.path.isfile(os.path.join(lead_path,f))], key=sort_key)
        lead_list = get_image(lead_path, file_names)
        # noisy_list = get_noisy_image(lead_list)

        result_path = f'./examples/examples_for_check_digitization/final_diagnosis/'  # 최종 진단 결과 저장 경로
        result_image_path = result_path + f'{dir_name}.png'  # 최종 진단 결과 저장 경로
        result_path_with_target = f'./examples/examples_for_check_digitization/final_diagnosis_with_target/'  # 최종 진단 결과 저장 경로
        result_image_path_with_target = result_path_with_target + f'{dir_name}.png'  # 최종 진단 결과 저장 경로
        create_directory_if_empty(result_path)
        create_directory_if_empty(result_path_with_target)

        extracted_with_paper_path = f'./examples/examples_for_check_digitization/extracted_with_paper/{dir_name}/'
        create_directory_if_empty(extracted_with_paper_path)

        signals, lead_2 = Digitization(lead_list)

        # signals.append(lead_2)
        # draw_extracted_on_paper(signals, lead_list, extracted_with_paper_path)

        # ECG classification
        result, extracted_signals = Classifier(signals, lead_2, class_num, lead_len=lead_len, lead_2_len=lead_2_len,
                                               DEVICE=DEVICE)
        result_target, raw_signals = ClassifierRaw(dir_name, class_num, lead_len=lead_len, lead_2_len=lead_2_len,
                                                   DEVICE=DEVICE)

        # Plot outputs of the model
        plot_results(result.cpu().numpy(),
                     extracted_signals.squeeze(0).cpu().numpy(),
                     np.expand_dims(lead_2, axis=0),
                     diagnosis_classes,
                     result_image_path)  # 이미지 추출 신호로부터 진단 시긱화

        plot_results_with_target(result.cpu().numpy(),
                                 result_target.cpu().numpy(),
                                 extracted_signals.squeeze(0).cpu().numpy(),
                                 raw_signals.squeeze(0).cpu().numpy(),
                                 diagnosis_classes,
                                 result_image_path_with_target)  # 이미지 추출 신호, 원본 신호로부터 진단 시각화

        extracted_results.append(result.cpu().numpy())
        target_results.append(result_target.cpu().numpy())

    save_data(diagnosis_classes, data_list=directories, extracted_result_list=extracted_results,
              target_result_list=target_results, dir=data_dir, filename="diagnosis_results.xlsx")

    
print(f"{time.time() - start:.5f} sec")  # CPU : 48.6초 소요