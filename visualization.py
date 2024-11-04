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


# from fakeguard.guard3 import src_image_path


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
        return None

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

    image_path = '/home/user/bob/fakeguard/image'
    image_sets = ['Karina1', 'Karina2', 'Karina3', 'Karina4', 'Karina5', 'sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'pm', 'pm2', 'pm3', 'pm4', 'pm5', 'IU', 'yua', 'Timothee']
    tsv_path = '/home/user/bob/2fakeguard/similarity/A_embeddings.tsv'
    meta_path = '/home/user/bob/2fakeguard/similarity/A_metadata.tsv'
    final_path = '/home/user/bob/fakeguard/output/final_image.png'
    
        # 보호된 이미지(노이즈 삽입된) 까지 비교하기
    # vec1 = extract_embedding(mtcnn(Image.open(final_path)))
    # vec2 = face_swap.extract_embedding(np.array(Image.open(final_path)))
    t3 = Image.open(final_path).convert('RGB')
    t3 = transform(t3).unsqueeze(0).to(device)
    vec3 = netArc(t3).flatten()
    with open(tsv_path,"w") as t:
        # t.write("\t".join(map(str, vec1.tolist())) + "\n")
        # t.write("\t".join(map(str, vec2.tolist())) + "\n")
        t.write("\t".join(map(str, vec3.tolist())) + "\n")
    with open(meta_path, "w") as m:
        # m.write("F_FINAL\nW_FIANL\nA_FINAL")
        m.write("A_FINAL\n")
    
    #
    # Facenet 임베딩 벡터
    # for img in image_sets:
    #     vec = extract_embedding(mtcnn(Image.open(os.path.join(image_path, (img+".png")))))
    #     with open(tsv_path, "a") as t:
    #         t.write("\t".join(map(str, vec.tolist())) + "\n")
    #     with open(meta_path, "a") as m:
    #         m.write("F_" + img + "\n")
    #     print(f"save::: embedding vector of {img}")
    #
    # # webface 임베딩 벡터
    # for img in image_sets:
    #     vec = face_swap.extract_embedding(np.array(Image.open(os.path.join(image_path, (img + ".png")))))
    #     with open(tsv_path, "a") as t:
    #         t.write("\t".join(map(str, vec.tolist())) + "\n")
    #     with open(meta_path, "a") as m:
    #         m.write("W_" + img + "\n")
    #     print(f"save::: embedding vector of {img}")


    # arcface 임베딩 벡터
    for img in image_sets:
        tensor = Image.open(os.path.join(image_path, (img + ".png"))).convert('RGB')
        tensor = transform(tensor).unsqueeze(0).to(device)
        vec = netArc(tensor).flatten()
        with open(tsv_path, "a") as t:
            t.write("\t".join(map(str, vec.tolist())) + "\n")
        with open(meta_path, "a") as m:
            m.write("A_" + img + "\n")
        print(f"save::: embedding vector of {img}")





    # src_image_path = image_path + '/Karina1.png'
    # src_image = Image.open(src_image_path).convert('RGB')
    # src_tensor = transform(src_image).unsqueeze(0)
    # src_tensor = src_tensor.to(device)
    # adv_gen = netArc(src_tensor)
    # print(adv_gen)



if __name__ == '__main__':
    # FaceNet 모델과 MTCNN(face detector) 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)  # 단일 얼굴 탐지를 위해 keep_all=False
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # inswapper 모델 경로 설정
    inswapper = "/home/user/bob/model/inswapper_128.onnx"
    # FaceSwap 객체 생성
    face_swap = FaceSwap(inswapper)

    # arcface 모델
    netArc_checkpoint = "/home/user/bob/fakeguard/arcface_checkpoint.tar"
    netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
    netArc = netArc_checkpoint
    netArc = netArc.to(device)
    netArc.eval()

    device = "cuda"
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # w600k 모델의 경우 입력 크기가 112x112일 수 있음
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
    ])

    main()

