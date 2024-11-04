import sys 
import cv2
import numpy as np
import math
from skimage.transform import resize

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def resize_image(image, target_shape):
    return resize(image, target_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_image> <target_image>")
        sys.exit(1)

    source_image_path = sys.argv[1]
    target_image_path = sys.argv[2]

    # 이미지 읽기
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
    
    # 소스와 타겟 이미지 크기가 다를 경우, 타겟 이미지를 소스 이미지 크기로 조정
    if source_image.shape != target_image.shape:
        target_image = resize_image(target_image, source_image.shape)
    
    psnr = calculate_psnr(source_image, target_image)
    print(f"PSNR: {psnr}")
    
if __name__ == '__main__':
    main()