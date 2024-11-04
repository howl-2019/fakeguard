import sys
from PIL import Image
import numpy as np
from facedetector import FaceSwap  # facedetector 파일에서 FaceSwap 클래스 import
from scipy.spatial.distance import cosine
import torch

# 유클리드 거리 기반 유사도 계산
def euclidean_similarity(source, target):
    source = torch.tensor(source, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    distance = torch.dist(source, target, p=2)
    similarity = 1 / (1 + distance)  # 유사도를 0과 1 사이로 정규화
    return similarity

def main():
    # 명령줄에서 두 이미지 파일을 입력받음
    if len(sys.argv) < 3:
        print("Usage: python script.py <source_image> <target_image>")
        sys.exit(1)

    source_image_path = sys.argv[1]
    target_image_path = sys.argv[2]

    # 모델 경로 설정 (이미 사용 중인 모델 경로 사용)
    model_path = "./inswapper_128.onnx"
    
    # FaceSwap 객체 생성
    face_swap = FaceSwap(model_path)

    # 소스 및 타겟 이미지 열기
    source_image = Image.open(source_image_path)
    target_image = Image.open(target_image_path)

    # 임베딩 추출
    source_embedding = face_swap.extract_embedding(np.array(source_image))
    target_embedding = face_swap.extract_embedding(np.array(target_image))

    print(source_embedding)
    
    if source_embedding is None or target_embedding is None:
        print("One of the images did not contain a detectable face.")
        sys.exit(1)

    # 코사인 유사도 계산
    similarity = 1 - cosine(source_embedding, target_embedding)
    print(f"Cosine similarity: {similarity}")

    # 유클리드 거리 유사도 계산
    similarity = euclidean_similarity(source_embedding, target_embedding)
    print(f"Euclidean Similarity - inswapper embedding: {similarity.item()}")

if __name__ == '__main__':
    main()

