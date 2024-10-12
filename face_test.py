import onnx
import torch
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms

# 1. ONNX 모델 로드
onnx_model_path = '../model/w600k_r50.onnx'
onnx_model = onnx.load(onnx_model_path)

# 2. ONNX 모델을 PyTorch 모델로 변환
torch_model = convert(onnx_model)

# 3. PyTorch 모델을 평가 모드로 설정
torch_model.eval()

# 4. 이미지 전처리 함수 정의
def preprocess_image(image_path):
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')

    # 이미지 전처리 (모델의 입력 형식에 맞게 크기와 정규화 수행)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),   # w600k 모델의 경우 입력 크기가 112x112일 수 있음
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원을 추가하여 (1, 3, 112, 112) 형태로 만듦
    return image_tensor

image_path = 'image/bboriginal.png'  # 처리할 이미지 경로
input_tensor = preprocess_image(image_path)

embedding_vector = torch_model(input_tensor)
print(embedding_vector)
