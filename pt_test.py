import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

class FaceSwap:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.face_swapper = self.load_model()  # 모델 로드

    def load_model(self):
        # 모델 구조 정의
        model = YourModelClass()  # YourModelClass를 적절한 모델 클래스로 변경하세요.
        model.load_state_dict(torch.load(self.model_path))  # 상태 사전 로드
        model.eval()  # 평가 모드로 설정
        return model

    def preprocess_image(self, image):
        # OpenCV 이미지를 PIL 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),  # 모델의 입력 크기에 맞게 조정
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return preprocess(image)

    def get_one_face(self, frame):
        # 얼굴 감지를 수행하고 얼굴 박스의 좌표를 반환
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            return boxes[np.argmin(boxes[:, 0])]
        return None

    def extract_face_embedding(self, source_image_path, target_image_path):
        # 이미지 경로를 사용하여 이미지를 읽어옵니다.
        src_image = cv2.imread(source_image_path)
        dst_image = cv2.imread(target_image_path)
        
        # 소스 및 대상 이미지를 전처리
        src_tensor = self.preprocess_image(src_image).unsqueeze(0)  # 배치 차원 추가
        dst_tensor = self.preprocess_image(dst_image).unsqueeze(0)  # 배치 차원 추가

        # 디버깅: 소스 및 대상 텐서의 모양 출력
        print(f"Source tensor shape: {src_tensor.shape}")
        print(f"Target tensor shape: {dst_tensor.shape}")

        # 모델을 통해 임베딩 추출
        with torch.no_grad():
            embedding = self.face_swapper(src_tensor, dst_tensor)

        # 임베딩을 numpy 배열로 변환
        embedding_np = embedding.cpu().numpy().flatten()

        return embedding_np

# 사용 예
model_path = '../model/inswapper.pt'  # 모델 파일 경로
face_swapper = FaceSwap(model_path)

# 이미지 경로
source_image_path = 'original.png'
target_image_path = 'animation.png'

# 얼굴 임베딩 추출
embedding = face_swapper.extract_face_embedding(source_image_path, target_image_path)
print("Extracted embedding:", embedding)
