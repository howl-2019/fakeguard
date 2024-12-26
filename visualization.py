import sys
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import argparse
import numpy as np
import torch
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
import torchvision.transforms as transforms
import cv2


from torch.utils.tensorboard import SummaryWriter
import pandas as pd
    

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

#facenet
def extract_embedding(face):
    if face is not None:
        # 임베딩 벡터 추출
        face = face.unsqueeze(0).to(device)  # 배치 차원 추가
        embedding = model(face)
        embedding_vector = embedding.detach().cpu().numpy().flatten()

    else:
        print("얼굴을 찾을 수 없습니다.")
        return None

    # print(type(embedding_vector))
    return embedding_vector



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
    
    # print(":::::bbox::::::", list(x1, y1, x2, y2))

    return cropped_face


def main():
    # 인자 처리
    # if len(sys.argv) < 3:
    #     print("Usage: python script.py <source_image> <target_image>")
    #     sys.exit(1)

    # source_image_path = sys.argv[1]
    # target_image_path = sys.argv[2]

    # 이미지 읽기
    # source_image = Image.open(source_image_path)
    # target_image = Image.open(target_image_path)

    # image_sets = ['Karina1', 'Karina2', 'Karina3', 'Karina4', 'Karina5', 'sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'pm', 'pm2', 'pm3', 'pm4', 'pm5', 'IU', 'yua', 'Timothee']
    
    # final_path = '/home/user/bob/fakeguard/output/final_image.png'
    
    
    

    
    
    # Facenet 임베딩 벡터
    for img in os.listdir(image_dir):
        if img.endswith("png"):
            if img.startswith("eps"): continue
            image = Image.open(os.path.join(image_dir, img))
            face = mtcnn(image.convert('RGB'))
            if face is None: continue
            vec = extract_embedding(face)
            if vec is None: continue
            with open(F_tsv_path, "a") as t:
                t.write("\t".join(map(str, vec.tolist())) + "\n")
            with open(F_meta_path, "a") as m:
                m.write(img + "\n")
            print(f"save::: embedding vector of {img} by FaceNet")
            

    # webface 임베딩 벡터
    for img in os.listdir(image_dir):
        if img.endswith("png"):
            if img.startswith("eps"): continue
            image = process_images(os.path.join(image_dir, img))
            if image is None: continue
            vec = face_swap.extract_embedding(np.array(image))
            if vec is None: continue
            with open(W_tsv_path, "a") as t:
                t.write("\t".join(map(str, vec.tolist())) + "\n")
            with open(W_meta_path, "a") as m:
                m.write(img + "\n")
            print(f"save::: embedding vector of {img} by WebFace")


    # arcface 임베딩 벡터
    for img in os.listdir(image_dir):
        if img.endswith("png"):
            if img.startswith("eps"): continue
            # tensor = Image.open(os.path.join(image_dir, img)).convert('RGB')
            image = process_images(os.path.join(image_dir, img))
            if image is None: continue
            tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
            vec = netArc(tensor).flatten()
            if vec is None: continue
            with open(A_tsv_path, "a") as t:
                t.write("\t".join(map(str, vec.tolist())) + "\n")
            with open(A_meta_path, "a") as m:
                m.write(img + "\n")
            print(f"save::: embedding vector of {img} by ArcFace")
    





def visualization():
    print("\n\nwriting TensorBoard\n")
    
    # arcface 벡터 파일 불러오기
    vectors_A = pd.read_csv(A_tsv_path, sep='\t', header=None).values
    # meta_auto_df = pd.read_csv(meta_auto_path, sep='\t', header=None).values
    with open(A_meta_path, 'r') as f:
        meta = [line.strip() for line in f]
    # TensorBoard SummaryWriter 생성 (log_dir 설정)
    writer_A = SummaryWriter(os.path.join(vec_dir,'Arcface'))
    # PyTorch Tensor로 변환 후 임베딩 추가
    # embedding_auto_tensor = torch.tensor(vectors_auto, dtype=torch.float)
    writer_A.add_embedding(vectors_A, metadata=meta, tag="Embedding Vectors by Arcface")
    writer_A.close()
    
    
    # facenet 벡터 파일 불러오기
    vectors_F = pd.read_csv(F_tsv_path, sep='\t', header=None).values
    # meta_auto_df = pd.read_csv(meta_auto_path, sep='\t', header=None).values
    with open(F_meta_path, 'r') as f:
        meta = [line.strip() for line in f]
    # TensorBoard SummaryWriter 생성 (log_dir 설정)
    writer_F = SummaryWriter(os.path.join(vec_dir,'FaceNet'))
    # PyTorch Tensor로 변환 후 임베딩 추가
    # embedding_auto_tensor = torch.tensor(vectors_auto, dtype=torch.float)
    writer_F.add_embedding(vectors_F, metadata=meta, tag="Embedding Vectors by FaceNet")
    writer_F.close()
    
    
    
    # webface 벡터 파일 불러오기
    vectors_W = pd.read_csv(W_tsv_path, sep='\t', header=None).values
    # meta_auto_df = pd.read_csv(meta_auto_path, sep='\t', header=None).values
    with open(W_meta_path, 'r') as f:
        meta = [line.strip() for line in f]
    # TensorBoard SummaryWriter 생성 (log_dir 설정)
    writer_W = SummaryWriter(os.path.join(vec_dir,'WebFace'))
    # PyTorch Tensor로 변환 후 임베딩 추가
    # embedding_auto_tensor = torch.tensor(vectors_auto, dtype=torch.float)
    writer_W.add_embedding(vectors_W, metadata=meta, tag="Embedding Vectors by WebFace")
    writer_W.close()


if __name__ == '__main__':
    # FaceNet 모델과 MTCNN(face detector) 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)  # 단일 얼굴 탐지를 위해 keep_all=False
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
    
    
    image_dir = '/home/user/bob/fakeguard/analysis_result/pm'
    vec_dir = '/home/user/bob/fakeguard/vec/pm'
    
    A_tsv_path = os.path.join(image_dir, "A_embeddings.tsv")
    A_meta_path = os.path.join(image_dir, "A_metadata.tsv")
    F_tsv_path = os.path.join(image_dir, "F_embeddings.tsv")
    F_meta_path = os.path.join(image_dir, "F_metadata.tsv")
    W_tsv_path = os.path.join(image_dir, "W_embeddings.tsv")
    W_meta_path = os.path.join(image_dir, "W_metadata.tsv")

    # 모든 경로들을 리스트로 정리
    file_paths = [A_tsv_path, A_meta_path, F_tsv_path, F_meta_path, W_tsv_path, W_meta_path]

    # 파일 생성 반복문
    for path in file_paths:

        # 파일 생성 (내용 없이 빈 파일로 생성)
        with open(path, 'w') as f: pass
        
    main()

    visualization()
    
    