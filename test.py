import onnx

# ONNX 모델 파일 로드
model_path = '../model/inswapper_128.onnx'  # 모델 파일 경로를 지정하세요.
model = onnx.load(model_path)

# 모델의 그래프 구조 출력
for input_tensor in model.graph.input:
    print(f'Input name: {input_tensor.name}')
    shape = input_tensor.type.tensor_type.shape.dim
    shape_list = [dim.dim_value for dim in shape]
    print(f'Shape: {shape_list}')  # 입력 텐서의 크기 출력