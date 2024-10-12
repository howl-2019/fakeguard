import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import onnx
import onnxruntime
from torchvision import transforms
from PIL import Image

class FaceEmbeddingExtractor:
    def __init__(self, model_path: str):
        # ONNX 모델 로드
        self.model_path = model_path
        self.model = onnxruntime.InferenceSession(self.model_path)
        
        # MTCNN을 사용하여 얼굴 감지
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 이미지 전처리 설정
        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract_face_embedding(self, image_path: str):
        # 이미지 로드
        image = Image.open(image_path)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # BGR로 변환
        
        # 얼굴 감지
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            # 가장 첫 번째 얼굴 선택
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face = image[y1:y2, x1:x2]

            # 얼굴을 PIL 이미지로 변환
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # RGB로 변환
            
            # 얼굴 전처리
            face_tensor = self.preprocess(face_pil).unsqueeze(0)  # 배치 차원 추가
            
            # ONNX 모델을 통한 얼굴 벡터 추출
            input_data = {
                'target': face_tensor.numpy()  # 'target' 입력
            }
            # 'source' 입력은 임베딩을 위한 자리
            source_embedding = np.zeros((1, 512), dtype=np.float32)  # 초기화

            # 모델 추론
            output = self.model.run(None, {**input_data, 'source': source_embedding})

            # 임베딩 벡터를 반환
            return output[0]  # 모델의 첫 번째 출력 (임베딩 벡터)

        return None  # 얼굴이 감지되지 않은 경우

# 사용 예시
if __name__ == "__main__":
    model_path = '../model/inswapper_128.onnx'  # 모델 경로 설정
    image_path = 'original.png'   # 얼굴 이미지를 설정
    
    extractor = FaceEmbeddingExtractor(model_path)
    embedding = extractor.extract_face_embedding(image_path)
    
    if embedding is not None:
        print(f"Extracted embedding: {embedding}")
    else:
        print("No face detected in the image.")
