import torch
from onnx2torch import convert
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnx

class FaceSwap:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        onnx_model = onnx.load(self.model_path)
        model = convert(onnx_model)
        model.eval()
        
        return model

    def preprocess_image(self, image_tensor):
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Input should be a tensor.")

        # 텐서를 float32로 변환
        image_tensor = image_tensor.float()

        # 텐서가 4채널(RGBA)일 경우, 알파 채널을 제거하여 3채널(RGB)로 변환
        if image_tensor.shape[0] == 4:
            image_tensor = image_tensor[:3, :, :]  # RGBA의 A 채널을 제거하여 RGB로 변환

        # 배치 차원을 추가하여 (N, C, H, W) 형식으로 변환
        if image_tensor.ndimension() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # 이미지 값이 0~1 사이에 있도록 정규화 (assuming input range is [0, 255])
        image_tensor = image_tensor / 255.0

        # 이미지 전처리 (크기 조정 및 정규화 수행)
        transform = transforms.Compose([
            transforms.Resize((112, 112)),  # 모델의 입력 크기에 맞게 크기 조정
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
        ])

        # 크기 조정 및 정규화 적용
        processed_tensor = F.interpolate(image_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        processed_tensor = transform(processed_tensor.squeeze(0)).unsqueeze(0)  # 배치 차원을 제거 후 다시 추가

        return processed_tensor

    def extract_face_embedding(self, source_tensor):
        # 입력 텐서를 전처리
        input_tensor = self.preprocess_image(source_tensor)
        self.model = self.model.to('cuda')
        # 얼굴 임베딩 벡터 추출
        embedding_vector = self.model(input_tensor)

        return embedding_vector
