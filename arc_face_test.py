import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

image_path = "image/pm.png"
device = "cuda"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToTensor(),           # 이미지를 텐서로 변환
    transforms.Resize((112, 112)),   # ArcFace 모델의 입력 크기에 맞게 조정 (예시)
    transforms.Normalize(            # 이미지 정규화
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
image_tensor = image_tensor.to(device)

netArc_checkpoint = "arcface_checkpoint.tar"
netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
netArc = netArc_checkpoint
netArc = netArc.to(device)
netArc.eval()

latend_id = netArc(image_tensor)


print(latend_id)
np.savetxt('pm.txt', latend_id.detach().cpu().numpy(), fmt='%f')