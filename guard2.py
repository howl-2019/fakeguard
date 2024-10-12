import sys
import argparse
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from torchvision import transforms
import onnx
from onnx2torch import convert
import torchvision.transforms as transforms
import torch

device = "cuda"
eps = 0.1
onnx_model_path = '../model/w600k_r50.onnx'
onnx_model = onnx.load(onnx_model_path)

# 2. ONNX 모델을 PyTorch 모델로 변환
torch_model = convert(onnx_model)
print(torch_model)

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

def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img

def generate_one(src_image_path, target_image_path):
    if not os.path.exists(src_image_path):
        raise FileNotFoundError(f"Source image not found: {src_image_path}")
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    # source_tensor = img2tensor(self.transform(Image.open(src_image_path))).to(device)
    # target_tensor = img2tensor(self.transform(Image.open(target_image_path))).to(device)
    source_tensor = preprocess_image(src_image_path)
    target_tensor = preprocess_image(target_image_path)

    source_tensor.requires_grad_()
    target_tensor.requires_grad_()

    target_latent = torch_model(target_tensor)

    modifier = torch.zeros_like(source_tensor, requires_grad=True)

    t_size = 1000
    max_change = eps / 0.5  # scale from 0,1 to -1,1
    step_size = max_change

    for i in range(t_size):
        actual_step_size = step_size - (step_size - step_size / 100) / t_size * i

        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1) # adv, modifier connect
        adv_latent = torch_model(adv_tensor)
        
        with torch.enable_grad():
            #loss = (adv_latent - target_latent).norm()
            loss = 0.5 * torch.abs(adv_latent - target_latent).sum() + 0.5 * torch.norm(adv_latent - target_latent)
            #loss = torch.nn.functional.kl_div(adv_latent.log_softmax(dim=-1), target_latent, reduction='batchmean')            
            #loss = torch.nn.functional.mse_loss(adv_latent, target_latent)
            #loss = 1 - torch.nn.functional.cosine_similarity(adv_latent, target_latent, dim=0).mean()
            tot_loss = loss.sum()

        if not tot_loss.requires_grad:
            tot_loss = tot_loss.requires_grad_()
        
        grad = torch.autograd.grad(tot_loss, modifier, allow_unused=False)[0]
        if grad is None:
            print("skip")
            continue

        modifier = modifier - torch.sign(grad) * actual_step_size
        modifier = torch.clamp(modifier, -max_change, max_change)

        if i % 50 == 0:
            print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")

            iter_img = tensor2img(modifier + source_tensor)
            iter_img.save(f"modifier_image_iter_{i}.png")

            # Save the difference
            diff = torch.abs(modifier)
            diff_img = tensor2img(diff)
            diff_img.save(f"modifier_diff_iter_{i}.png")

    final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

    final_img = tensor2img(final_adv_batch)
    final_img.save("final_image.png")

    final_modifier_img = tensor2img(modifier + source_tensor)
    final_modifier_img.save("modifier_image_final.png")

    final_diff = torch.abs(modifier)
    final_diff_img = tensor2img(final_diff)
    final_diff_img.save("modifier_diff_final.png")


src_image_path = 'image/bboriginal.png'
target_image_path = 'image/bbanimation.png'

# 경로 문자열을 사용하여 generate_one 메서드 호출
generate_one(src_image_path, target_image_path)

print("Image processing completed!")