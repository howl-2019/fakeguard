import cv2
import sys
import numpy as np
from skimage.transform import resize
from numba import jit

def resize_image(image, target_shape):
    return resize(image, target_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

@jit
def mse(imageA:np.ndarray, imageB:np.ndarray):
    # 두 이미지의 차이를 계산
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

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
    
    ratio = mse(source_image, target_image)
    print(f"MSE(0에 가까울 수록 유사) : {ratio}")

if __name__ == "__main__":
    main()
    