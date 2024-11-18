import cv2
from parallel_shadow import add_parallel_light
from spot_shadow import add_spot_light

"""
Make shadowed image using main function
"""

def main():
    image_path = './shadow/picture.jpg'  # 처리할 이미지 경로
    spot_outpath = './shadow/picture_spt.jpg'  # 결과 이미지를 저장할 경로
    parl_outpath = './shadow/picture_prl.jpg'

    spot_result = add_spot_light(image_path)
    parl_result = add_parallel_light(image_path)

    if spot_result is not None:
        cv2.imwrite(spot_outpath, spot_result)
        print(f"그림자가 포함된 이미지가 저장되었습니다: {spot_outpath}")
    else:
        print("이미지를 처리하는 동안 오류가 발생했습니다.")

    if parl_result is not None:
        cv2.imwrite(parl_outpath, parl_result)
        print(f"그림자가 포함된 이미지가 저장되었습니다: {parl_outpath}")
    else:
        print("이미지를 처리하는 동안 오류가 발생했습니다.")

if __name__ == "__main__":
    main()