import torch
import onnx
from onnx2pytorch import ConvertModel

# ONNX 모델 파일 경로
onnx_model_path = 'w600k_r50.onnx'

# ONNX 모델을 불러오기
onnx_model = onnx.load(onnx_model_path)

# ONNX 모델을 PyTorch로 변환
pytorch_model = ConvertModel(onnx_model)

# PyTorch 모델 파일 경로
pytorch_model_path = 'w600k_r50.pt'

# 변환된 PyTorch 모델을 저장
torch.save(pytorch_model.state_dict(), pytorch_model_path)

print(f"PyTorch 모델이 {pytorch_model_path}에 저장되었습니다.")


