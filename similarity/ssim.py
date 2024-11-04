from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import numpy as np
import sys

def resize_image(image, target_shape):
    return resize(image, target_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

def main():
    # 명령줄에서 두 이미지 파일을 입력받음
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_image> <target_image>")
        sys.exit(1)

    source_image_path = sys.argv[1]
    target_image_path = sys.argv[2]

    # 소스 및 타겟 이미지 열기
    source_image = io.imread(source_image_path)
    target_image = io.imread(target_image_path)
    
    # 소스와 타겟 이미지 크기가 다를 경우, 타겟 이미지를 소스 이미지 크기로 조정
    if source_image.shape != target_image.shape:
        target_image = resize_image(target_image, source_image.shape)

    # SSIM 계산 (multichannel=True)
    score = ssim(source_image, target_image, win_size=3, channel_axis = -1) # multichannel=True)
    print(f'SSIM Score: {score}')


if __name__ == '__main__':
    main()
