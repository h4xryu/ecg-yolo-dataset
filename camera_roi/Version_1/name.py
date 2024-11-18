import numpy as np
import os
# Load the text file into a NumPy array
# Replace 'your_file.txt' with your file path
# path = os.path.join('e:\\data_set_yolov7\\output\\Version_1','0310','MR060001-0.txt')


directory = os.path.join('e:\\data_set_yolov7', "output","Version_1", "0311" )
os.makedirs(directory, exist_ok=True)
file_list = []


if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_name)
            file_list.sort()



for file_name in file_list:
    print(file_name.split('.'))
    # print(f"{directory}\\{file_name.split('.')[0]+63}.jpg")
    # os.rename(f"{directory}\\{file_name.split('.')[0]}.jpg", f"{directory}\\{file_name.split('.')[0]+63}.jpg")
