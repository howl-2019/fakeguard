import sys
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import torch
import cv2
import math
from skimage.transform import resize
import os
from insightface.app import FaceAnalysis
import insightface
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import re




def calculate_psnr(img1, img2):
    
    # 소스와 타겟 이미지 크기가 다를 경우, 타겟 이미지를 소스 이미지 크기로 조정
    if img1.shape != img2.shape:
        img2 = resize_image(img2, img1.shape)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def resize_image(image, target_shape):
    return resize(image, target_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)


# webface
class FaceSwap:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.face_analyser = self.get_face_analyser()
        self.face_swapper = self.get_face_swap_model()

    def get_face_analyser(self, providers=None, det_size=(320, 320)):
        face_analyser = FaceAnalysis(providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        return face_analyser

    def get_face_swap_model(self):
        model = insightface.model_zoo.get_model(self.model_path)
        return model

    def get_one_face(self, frame: np.ndarray):
        faces = self.face_analyser.get(frame)
        try:
            return min(faces, key=lambda x: x.bbox[0])
        except ValueError:
            return None

    def extract_embedding(self, image: np.ndarray):
        face = self.get_one_face(image)
        if face is not None:
            return face.embedding
        print("webface - 얼굴 없음")
        return None

def get_face_bbox(image):
    # 얼굴 감지 및 바운딩 박스 가져오기
    faces = face_analyser.get(image)
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴을 기준으로 함
    return max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
# 이미지에서 얼굴 바운딩 박스만 잘라내기
def process_images(source_img_path):
    source_image = cv2.imread(source_img_path)

    source_face = get_face_bbox(source_image)

    if source_face is None:
        print("Image does not contain a detectable face.")
        return None
    
    x1, y1, x2, y2 = source_face.bbox.astype(int)
    cropped_face = source_image[y1:y2, x1:x2]
    
    return cropped_face


# Function to extract numeric values from file names for sorting, including float numbers
def numerical_sort(value):
    numbers = re.findall(r'\d+\.\d+|\d+', value)  # Find float or integer numbers
    return float(numbers[0]) if numbers else float('inf')

def main():
    
    original_path = '/home/user/bob/fakeguard/image/pm.png'
    image_dir = '/home/user/bob/fakeguard/analysis_result/pm/eps'
    
    # 1. 비교 이미지 설정 
    original = process_images(original_path)
    origin_tensor = transform(Image.fromarray(original)).unsqueeze(0).to(device)
    origin_vec = netArc(origin_tensor).flatten()
    origin_vec = origin_vec.detach().cpu().numpy()
    
    # cos1 = []
    # cos1.append(1 - cosine(origin_vec, origin_vec))

    psnr = []
    psnr.append(calculate_psnr(original, original))

    
    ## noise eps 별 for문
    
    image_files = os.listdir(image_dir)
    sorted_image_files = sorted(image_files, key=numerical_sort)
    x_labels = [re.findall(r'\d+\.\d+|\d+', img)[0] for img in sorted_image_files]
    x_labels.insert(0, 'original') # 첫 원소: 원본 이미지와 비교 결과 
    
    print(x_labels)
    
    for img in sorted_image_files:
        if img.startswith("eps") and img.endswith("png"):
            print(f"{img} is processing")
            # 2. 얼굴 bbox 추출
            image = process_images(os.path.join(image_dir, img))
            psnr.append(calculate_psnr(original, image))

            # if image is None: 
            #     print(f"{img} does not have face")
            #     cos1.append(0)
            #     continue
            # tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
            # vec = netArc(tensor).flatten()
            # vec = vec.detach().cpu().numpy()
            # if vec is None:
            #     print(f"cannot extract vectors from {img}")
            #     cos1.append(0)
            #     continue
            # cos1.append(1 - cosine(origin_vec, vec))
            
            # psnr.append(calculate_psnr(original, image))
            
            
    # # 3. 원본 이미지 vs noise 이미지 cosine 유사도 계산
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_labels, cos1, marker='o', linestyle='-', color='b')
    # plt.xlabel('Noise EPS of Image')
    # plt.ylabel('Cosine Similarity')
    # plt.title('Cosine Similarity between Original and Noised Images by Arcface')
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.savefig('arc_cosine_similarity_plot.png')  # Save the plot as an image file
    
    
    
    # 4. 원본 이미지 vs noise 이미지 psnr 계산
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, psnr, marker='o', linestyle='-', color='b')
    plt.xlabel('Noise EPS of Image')
    plt.ylabel('PSNR')
    plt.title('PSNR between Original and Noised Images by Arcface (area: bbox)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('arc_psnr_plot.png')  # Save the plot as an image file
    
    # 5. 임계치 둘까? 그래프 확인? 
    
    
    
    # 3.1. 원본 이미지 vs noise+딥페이크 이미지 cosine 유사도 계산 
    
    d_image_dir = '/home/user/bob/fakeguard/analysis_result/pm/auto'
    d_original_path = '/home/user/bob/fakeguard/analysis_result/pm/auto_d_pm.png'


    # deep_image_files = os.listdir(deep_image_dir)
    d_image_files = [img for img in os.listdir(d_image_dir) if img.startswith('auto_d_eps_')]
    d_sorted_image_files = sorted(d_image_files, key=numerical_sort)
    
    d_x_labels = [re.findall(r'\d+\.\d+|\d+', img)[0] for img in d_sorted_image_files]
    d_x_labels.insert(0, 'd_original') # 첫 원소: 원본 이미지와 비교 결과 
    
    d_original = process_images(d_original_path)
    d_origin_tensor = transform(Image.fromarray(d_original)).unsqueeze(0).to(device)
    d_origin_vec = netArc(d_origin_tensor).flatten()
    d_origin_vec = d_origin_vec.detach().cpu().numpy()
    cos2 = [1 - cosine(d_origin_vec, d_origin_vec)]
    
    for d_img in d_sorted_image_files:
        if img.endswith("png"):
            print(f"{d_img} is processing")
            # 2. 얼굴 bbox 추출
            d_image = process_images(os.path.join(d_image_dir, d_img))
            if d_image is None: 
                print(f"{d_img} does not have face")
                continue
            d_tensor = transform(Image.fromarray(d_image)).unsqueeze(0).to(device)
            d_vec = netArc(d_tensor).flatten()
            d_vec = d_vec.detach().cpu().numpy()
            if d_vec is None:
                print(f"cannot extract vectors from {d_img}")
                cos2.append(0)
                continue
            cos2.append(1 - cosine(d_origin_vec, d_vec))
            
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(d_x_labels, cos2, marker='o', linestyle='-', color='b')
    plt.xlabel('Noise EPS of Image')
    plt.ylabel('Cosine Similarity - Original vs Deepfakes')
    plt.title('Cosine Similarity between Original and Deepfake Noised Images by Arcface')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('arc_cosine_similarity_deepfake_plot.png')  # Save the plot as an image file
    
    
    
    
    
    
if __name__ == '__main__':
    
    # inswapper 모델 경로 설정
    inswapper = "/home/user/bob/model/w600k_r50.onnx"
    # FaceSwap 객체 생성
    face_swap = FaceSwap(inswapper)
    det_size=(320, 320)
    face_analyser = face_swap.get_face_analyser(det_size)
    
    
    # arcface 모델
    device = "cuda"
    netArc_checkpoint = "/home/user/bob/model/arcface_checkpoint.tar"
    netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
    netArc = netArc_checkpoint
    netArc = netArc.to(device)
    netArc.eval()

    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # w600k 모델의 경우 입력 크기가 112x112일 수 있음
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
    ])
    
    
    main()
