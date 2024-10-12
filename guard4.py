
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
from insightface.app import FaceAnalysis
import cv2
import torch.nn.functional as F
from torchvision.transforms.v2.functional import posterize_image

from opt import img2tensor

# PSNR 기반 손실 함수 정의
def psnr_loss(adv_tensor, source_tensor, max_pixel=1.0):
    mse = F.mse_loss(adv_tensor, source_tensor)
    return -10 * torch.log10(max_pixel**2 / mse)

# triplet loss function - 이건 노이즈의 한계를 안 정함
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, p=2) # L2 norm
    neg_dist = torch.norm(anchor - negative, p=2)
    pos_dist = pos_dist / (1 + pos_dist)
    neg_dist = neg_dist / (1 + neg_dist)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss, pos_dist, neg_dist

# 복합 손실 함수 정의
# def combined_loss(anchor, positive, negative, source_image, adv_image, margin=1.0, lambda_reg=0.1):
#     # 트리플렛 손실 (임베딩 유사성)
#     pos_dist = torch.norm(anchor - positive, p=2)
#     neg_dist = torch.norm(anchor - negative, p=2)
#     triplet_loss = torch.relu(pos_dist - neg_dist + margin)
#
#     # 시각적 일관성 손실 (노이즈가 추가된 이미지가 원본과 유사하도록)
#     visual_consistency_loss = F.mse_loss(adv_image, source_image)
#
#     # 총 손실 계산
#     total_loss = triplet_loss + lambda_reg * visual_consistency_loss
#     return total_loss

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((112, 112)),   # w600k 모델의 경우 입력 크기가 112x112일 수 있음
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
    ])

    image_tensor = transform(image).unsqueeze(0)  # 배치 차원을 추가하여 (1, 3, 112, 112) 형태로 만듦
    return image_tensor

# def crop_to_square(img):
#     size = 512
#     image_transforms = transforms.Compose(
#         [
#             transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(size),
#         ]
#     )
#     return image_transforms(img)

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

    # source_tensor = img2tensor(Image.open(src_image_path))
    # target_tensor = img2tensor(Image.open(target_image_path))

    source_tensor = preprocess_image(src_image_path)
    target_tensor = preprocess_image(target_image_path)

    source_tensor.requires_grad_()
    target_tensor.requires_grad_()

    source_latent = torch_model(source_tensor)
    target_latent = torch_model(target_tensor)

    # modifier = torch.zeros_like(source_tensor, requires_grad=True)
    # PSNR 쓰려면 mse 0이면 안 돼 -> 초기에 약간의 노이즈 삽입
    modifier = torch.randn_like(source_tensor, requires_grad=True) * 0.01
    modifier = modifier.detach().requires_grad_(True)

    optimizer = torch.optim.SGD([modifier], lr=0.01)


    t_size = 3000
    max_change = eps / 0.5  # scale from 0,1 to -1,1
    step_size = max_change

    for i in range(t_size):
        # Lambda 값 동적 감소
        # lambda_reg = max(0.1 - (i * 0.001), 0.01)
        # lambda_reg = 0.05
        lambda_reg = max(0.05 - (i * 0.001), 0.005)
        optimizer.zero_grad()

        actual_step_size = step_size - (step_size - step_size / 100) / t_size * i

        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1) # adv, modifier connect
        adv_latent = torch_model(adv_tensor)

        # print("Max pixel value in source_tensor:", source_tensor.max().item())
        # print("Max pixel value in adv_tensor:", adv_tensor.max().item())

        with torch.enable_grad():
            # triplet loss
            # loss = triplet_loss(adv_latent, target_latent, torch_model(source_tensor))
            # loss = triplet_loss(adv_latent, target_latent, source_latent) + lambda_reg * F.mse_loss(adv_tensor, source_tensor)

            triplet_loss_value, pos_dist, neg_dist = triplet_loss(adv_latent, target_latent, source_latent, 0.5)
            # loss = triplet_loss_value + lambda_reg * F.mse_loss(adv_tensor, source_tensor)
            psnr_loss_values = psnr_loss(adv_tensor, source_tensor)
            psnr_loss_values = psnr_loss_values / (psnr_loss_values.abs() + 1)
            loss = triplet_loss_value + lambda_reg * psnr_loss_values

            # 복합 손실 계산
            ## lambda_reg: 두 손실 간 가중치 보통 0.01~0.1 (유사도 중요: 작게. 비가시성 중요: 크게)
            # loss = combined_loss(adv_latent, target_latent, source_latent, source_tensor, adv_tensor, margin=1.0, lambda_reg=lambda_reg)

            #loss = (adv_latent - target_latent).norm()
            # loss = 0.5 * torch.abs(adv_latent - target_latent).sum() + 0.5 * torch.norm(adv_latent - target_latent)
            #loss = torch.nn.functional.kl_div(adv_latent.log_softmax(dim=-1), target_latent, reduction='batchmean')
            #loss = torch.nn.functional.mse_loss(adv_latent, target_latent)
            #loss = 1 - torch.nn.functional.cosine_similarity(adv_latent, target_latent, dim=0).mean()
            tot_loss = loss.sum()

            tot_loss.backward(retain_graph=True)

        # if not tot_loss.requires_grad:
        #     tot_loss = tot_loss.requires_grad_()

        # grad = torch.autograd.grad(tot_loss, modifier, allow_unused=False)[0]
        grad = modifier.grad
        if grad is None:
            print("skip")
            continue

        optimizer.step()
        modifier.data = torch.clamp(modifier, -max_change, max_change)

        # modifier = modifier - torch.sign(grad) * actual_step_size
        # modifier = torch.clamp(modifier, -max_change, max_change)
        # modifier = torch.clamp(modifier, -eps, eps)

        # modifier.grad.zero_()

        if i % 50 == 0:
            print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}, \tPos Dist: {pos_dist.item():.3f}, '"
                  f"\tNeg Dist: {neg_dist.item():.3f} \tPSNR Loss: {psnr_loss_values:.3f}")

            iter_img = tensor2img(modifier + source_tensor)
            iter_img.save(f"triplet_modifier_image_iter_{i}.png")

            # Save the difference
            diff = torch.abs(modifier)
            diff_img = tensor2img(diff)
            diff_img.save(f"triplet_modifier_diff_iter_{i}.png")

    # final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

    final_img = tensor2img(modifier + source_tensor)
    # final_img = tensor2img(final_adv_batch)
    final_img.save("triplet_final_image.png")

    final_modifier_img = tensor2img(modifier + source_tensor)
    final_modifier_img.save("triplet_modifier_image_final.png")

    final_diff = torch.abs(modifier)
    final_diff_img = tensor2img(final_diff)
    final_diff_img.save("triplet_modifier_diff_final.png")

    return final_img

def get_face_analyser(det_size):
    face_analyser = FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_face_bbox(image):
    # 얼굴 감지 및 바운딩 박스 가져오기
    faces = face_analyser.get(image)
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴을 기준으로 함
    return max(faces, key=lambda x: x.bbox[2] * x.bbox[3])

def crop_face_bbox(image, face):
    # 얼굴 바운딩 박스 부분만 자르기
    x1, y1, x2, y2 = face.bbox.astype(int)
    cropped_face = image[y1:y2, x1:x2]

    return cropped_face

def process_images(source_img_path, target_img_path):
    source_image = cv2.imread(source_img_path)
    target_image = cv2.imread(target_img_path)

    source_face = get_face_bbox(source_image)
    target_face = get_face_bbox(target_image)

    if source_face is None or target_face is None:
        print("One of the images does not contain a detectable face.")
        return None
    source_bbox = source_face.bbox.astype(int).tolist()
    print(":::::bbox::::::", source_bbox)

    cropped_source_face = crop_face_bbox(source_image, source_face)
    cropped_target_face = crop_face_bbox(target_image, target_face)

    cv2.imwrite("source_face.png", cropped_source_face)
    cv2.imwrite("target_face.png", cropped_target_face)
    print("Cropped faces saved.")

    return source_bbox


def replace_image(original_image: np.ndarray, cropped_image, coords):
    # 좌표 가져오기
    x1, y1, x2, y2 = coords

    # 크롭된 이미지를 coords 크기에 맞게 변형
    target_height = y2 - y1
    target_width = x2 - x1
    quarter_h = target_height // 4
    remaining_h = target_height - quarter_h*2
    quarter_w = target_width // 4
    remaining_w = target_width - quarter_w*2

    if isinstance(cropped_image, Image.Image):
        # PIL 이미지를 NumPy 배열로 변환
        cropped_image = np.array(cropped_image)

    # 채널 순서 변환 (RGB -> BGR)
    if cropped_image.shape[-1] == 3:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

    # 크기 맞추기
    resized_cropped_image = cv2.resize(cropped_image, (target_width, target_height))

    # 데이터 타입 확인 및 변환 (float -> uint8)
    if resized_cropped_image.dtype != original_image.dtype:
        resized_cropped_image = resized_cropped_image.astype(original_image.dtype)


    # ver3. 알파 마스크 생성 (중심은 1, 모서리로 갈수록 0으로 감소) - 그럼 세로도 ㄱㄱ
    ## 가로세로 각 4등분해서 가운데 두 칸씩은 무조건 100% 투명도 보장
    mask_y_q = np.linspace(0, 1, quarter_h)
    mask_y = np.concatenate((mask_y_q, np.ones(remaining_h), mask_y_q[::-1]), axis=0)[:, np.newaxis]

    mask_x_q = np.linspace(0, 1, quarter_w)
    mask_x = np.concatenate((mask_x_q, np.ones(remaining_w), mask_x_q[::-1]), axis=0)[np.newaxis, :]

    mask = np.minimum(mask_y, mask_x)

    mask = mask[..., np.newaxis]  # 채널 차원 추가

    # 블렌딩 (마스크가 3채널로 변환되어야 함)
    mask = np.repeat(mask, 3, axis=2)

    # 원본 이미지에 다시 삽입 (크기가 정확히 맞는지 확인)
    if resized_cropped_image.shape[:2] == (target_height, target_width):
        blended_image = (resized_cropped_image * mask + original_image[y1:y2, x1:x2] * (1 - mask)).astype(
            original_image.dtype)
        cv2.imwrite("output/triplet_blended_image.png", blended_image)
        original_image[y1:y2, x1:x2] = blended_image
    else:
        print("Error: Resized cropped image dimensions do not match target dimensions.")

    return original_image


# main

device = "cuda"
eps = 0.04
onnx_model_path = '../model/w600k_r50.onnx'
onnx_model = onnx.load(onnx_model_path)
det_size=(320, 320)
face_analyser = get_face_analyser(det_size)

torch_model = convert(onnx_model)
torch_model.eval()

src_image_path = 'image/original.png'
target_image_path = 'image/IU.png'

src_bbox = process_images(src_image_path, target_image_path)

noised_img = generate_one("source_face.png", "target_face.png")

original_img = cv2.imread(src_image_path)
final_image = replace_image(original_img, noised_img, src_bbox)

cv2.imwrite("output/final_triplet_blended_image.png", final_image)

print("Image processing completed!")
