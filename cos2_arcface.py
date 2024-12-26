import os
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import insightface
import torchvision.transforms as transforms
import torch
from PIL import Image


import cv2
from tabulate import tabulate



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



def compare_images(image_paths):
    """
    Compare cosine similarity between multiple images and generate a table.
    """
    # 벡터 추출
    embeddings = []
    for path in image_paths:
        print(f"processing now::::: {path}")
        image = process_images(path)
        if image is None:
            print(f"{path} does not contain a detectable face.")
            embeddings.append(None)
            continue

        tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
        vec = netArc(tensor).flatten()
        vec = vec.detach().cpu().numpy()
        
        if vec is None:
            print(f"Cannot extract embedding from {path}.")
            embeddings.append(None)
        else:
            embeddings.append(vec)

    # 모든 쌍 코사인 유사도 비교
    n = len(image_paths)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if embeddings[i] is not None and embeddings[j] is not None:
                similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])
            else:
                similarity_matrix[i, j] = np.nan  # Use NaN for missing embeddings

    # 표 생성
    table = []
    headers = ["Image"] + [os.path.basename(path) for path in image_paths]

    for i, path in enumerate(image_paths):
        row = [os.path.basename(path)] + [
            f"{similarity_matrix[i, j]:.2f}" if not np.isnan(similarity_matrix[i, j]) else "N/A"
            for j in range(n)
        ]
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    # Example usage with file paths
    image_paths = [
        "/home/user/bob/fakeguard/image/pm.png",
        # "/home/user/bob/fakeguard/analysis_result/pm/auto_d_pm.png",

        "/home/user/bob/fakeguard/image/celeb/myeditoriginal.png",
        "/home/user/bob/fakeguard/image/celeb/myedit0.1.png",
        "/home/user/bob/fakeguard/image/celeb/myedit0.14.png",
        "/home/user/bob/fakeguard/image/celeb/myedit0.16.png",
        "/home/user/bob/fakeguard/image/celeb/myedit0.3.png",

        # "/home/user/bob/fakeguard/image/celeb/kebin+sm.png",
        # "/home/user/bob/fakeguard/analysis_result/sy/d_sy9.png",
        # "/home/user/bob/fakeguard/analysis_result/sy/sm5.png"

    ]

    # inswapper 모델 경로 설정
    inswapper = "/home/user/bob/model/w600k_r50.onnx"

    # FaceSwap 객체 생성
    face_swap = FaceSwap(inswapper)
    det_size = (320, 320)
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

    compare_images(image_paths)
