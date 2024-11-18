import numpy as np
import os
# Load the text file into a NumPy array
# Replace 'your_file.txt' with your file path
# path = os.path.join('e:\\data_set_yolov7\\output\\Version_1','0310','MR060001-0.txt')


directory = os.path.join('e:\\data_set_yolov7', "output","Version_4" )
os.makedirs(directory, exist_ok=True)
file_list = []


if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_name)
            file_list.sort()

# classes = ['lead_I','lead_II','lead_III','lead_aVR','lead_aVL','lead_aVF','lead_II_ext','lead_V1','lead_V2','lead_V3','lead_V4','lead_V5','lead_V6']
classes = [0,1,2,3,4,5,6,7,8,9,10,11,12]
output_directory = os.path.join('e:/data_set_yolov7',"Version_1","train","label")

for file_name in file_list:

    data = np.loadtxt(os.path.join(directory, file_name), dtype={'names': ('label', 'x', 'y', 'w', 'h'),
                                          'formats': ('U10', 'f4', 'f4', 'f4', 'f4')})

    center_x = data['x'] - data['w'] / 16 
    center_y = data['y'] - data['h'] / 16 

    # Add the center points as temporary fields for sorting
    data_with_centers = np.array(
        list(zip(data['x'], data['y'], data['w'], data['h'], center_x, center_y)),
        dtype={'names': ('x', 'y', 'w', 'h', 'center_x', 'center_y'),
            'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4')}
    )


    # Define a tolerance for "similar x values"
    x_tolerance = 0.1  # Adjust this threshold as needed

    # Sort data by x first
    sorted_by_x = np.sort(data_with_centers, order='center_x')

    # Sort within groups of similar x values
    sorted_indices = []
    start_idx = 0

    for i in range(1, len(sorted_by_x)):
        if abs(sorted_by_x['center_x'][i] - sorted_by_x['center_x'][start_idx]) > x_tolerance:
            # Sort the current group by y
            group = sorted_by_x[start_idx:i]
            sorted_indices.extend(np.argsort(group['center_y']) + start_idx)
            start_idx = i

    # Handle the last group
    group = sorted_by_x[start_idx:]
    sorted_indices.extend(np.argsort(group['center_y']) + start_idx)

    # Apply sorted indices
    sorted_data = sorted_by_x[np.array(sorted_indices)]
    sorted_data = sorted_data[['x', 'y', 'w', 'h']]

    data_with_center = np.array(
        list(zip(classes, sorted_data['x'], sorted_data['y'], sorted_data['w'], sorted_data['h'])),
        dtype={'names': ('label', 'x', 'y', 'w', 'h'),
            'formats': ('U10', 'f4', 'f4', 'f4', 'f4')}
    )
    print(data_with_center)


    output_txt_path = os.path.join(output_directory,f"{file_name.split('.')[0]}.txt")
    # Save sorted data back to file (optional)
    np.savetxt(f"e:/data_set_yolov7/Version_1/train/label/{file_name.split('.')[0]}.txt", data_with_center, fmt='%s %.6f %.6f %.6f %.6f',
                comments='')

    # print(sorted_data)