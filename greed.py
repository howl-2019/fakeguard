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
from insightface.app import FaceAnalysis
import cv2
import wandb
from torch.optim.lr_scheduler import StepLR
from torch import optim 
from itertools import permutations

import sys

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project-asdf",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

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

def generate_one(src_image_path, target_image_path_1, target_image_path_2, target_image_path_3, target_image_path_4, w_1, w_2, w_3):
    # if not os.path.exists(src_image_path):
    #     raise FileNotFoundError(f"Source image not found: {src_image_path}")
    # if not os.path.exists(target_image_path):
    #     raise FileNotFoundError(f"Target image not found: {target_image_path}")

    source_tensor = preprocess_image(src_image_path)
    target_tensor_1 = preprocess_image(target_image_path_1)
    target_tensor_2 = preprocess_image(target_image_path_2)
    target_tensor_3 = preprocess_image(target_image_path_3)
    target_tensor_4 = preprocess_image(target_image_path_4)

    source_tensor = source_tensor.to(device)
    target_tensor_1 = target_tensor_1.to(device)
    target_tensor_2 = target_tensor_2.to(device)
    target_tensor_3 = target_tensor_3.to(device)
    target_tensor_4 = target_tensor_4.to(device)

    with torch.no_grad():
        target_auto_1 = torch_model(target_tensor_1)
        target_auto_2 = torch_model(target_tensor_2)
        target_auto_3 = torch_model(target_tensor_3)
        target_auto_4 = torch_model(target_tensor_4)
        target_gan_1 = netArc(target_tensor_1)
        target_gan_2 = netArc(target_tensor_2)
        target_gan_3 = netArc(target_tensor_3)
        target_gan_4 = netArc(target_tensor_4)
    
    all_auto = torch.stack([
    target_auto_1, target_auto_2, target_auto_3, target_auto_4
])
    all_gan = torch.stack([
    target_gan_1, target_gan_2, target_gan_3, target_gan_4
])
    target_auto = torch.mean(all_auto, dim=0)
    target_gan = torch.mean(all_gan, dim=0)
    
    # np.savetxt('target.txt', target_auto.numpy(), fmt='%f')
    
    # with torch.no_grad():
    #     source_auto = torch_model(source_tensor)
    #     source_gen = netArc(source_tensor)
    
    # modifier = torch.zeros_like(source_tensor, requires_grad=True)
    modifier = torch.zeros_like(source_tensor[:, :1, :, :], requires_grad=True)
    optimizer = optim.Adam([modifier], lr=0.01)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    # max_change = eps / epss  # scale from 0,1 to -1,1
    max_change = eps
    step_size = max_change

    for i in range(t_size):
        # actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        # adv_tensor = modifier + source_tensor # adv, modifier connect
        adv_tensor = adv_tensor.to(device)
        adv_auto = torch_model(adv_tensor)
        adv_gan = netArc(adv_tensor)
        
        l1_dis_auto = torch.abs(adv_auto - target_auto).sum()
        l2_dis_auto = torch.norm(adv_auto - target_auto)
        dual_dis_auto = w_1 * torch.abs(adv_auto - target_auto).sum() + w_2 * torch.norm(adv_auto - target_auto)
        mse_loss_auto = torch.nn.functional.mse_loss(adv_auto, target_auto)
        
        l1_dis_gan = torch.abs(adv_gan - target_gan).sum()
        l2_dis_gan = torch.norm(adv_gan - target_gan)
        dual_dis_gan = w_1 * torch.abs(adv_gan - target_gan).sum() + w_2 * torch.norm(adv_gan - target_gan) 
        mse_loss_gan = torch.nn.functional.mse_loss(adv_gan, target_gan)
        
        loss_auto = w_1*l1_dis_auto + w_2*l2_dis_auto + w_3*mse_loss_auto
        loss_gan = w_1*l1_dis_gan + w_2*l2_dis_gan + w_3 * mse_loss_gan
        loss = -(loss_auto + w_3 * loss_gan)
        # loss = -loss_gan

        # wandb.log({"distance_auto": dual_dis_auto,"mse_auto": mse_loss_auto, 
        #            "distance_gan": dual_dis_gan, "mse_gan": mse_loss_gan,
        #            "Loss": loss})
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(torch_model.parameters()) + list(netArc.parameters()), max_norm=1.0)
        optimizer.step()
        # scheduler.step()
        modifier.data.clamp_(-max_change, max_change)
        
        
        # if i % 50 == 0:
        #     print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")

            # iter_img = tensor2img(modifier + source_tensor)
            # iter_img.save(f"modifier_image_iter_{i}.png")

            # Save the difference
            # diff = torch.abs(modifier)
            # diff_img = tensor2img(diff)
            # diff_img.save(f"modifier_diff_iter_{i}.png")
    print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")
    print(f"auto {l1_dis_auto.mean().item():.3f}, {l2_dis_auto.mean().item():.3f}, {mse_loss_auto.mean().item():.3f}, {w_1}, {w_2}, {w_3}")
    print(f"gan {l1_dis_gan.mean().item():.3f}, {l2_dis_gan.mean().item():.3f}, {mse_loss_gan.mean().item():.3f}")
    final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

    final_img = tensor2img(final_adv_batch)
    final_img.save("final_image.png")

    final_modifier_img = tensor2img(modifier + source_tensor)
    final_modifier_img.save("modifier_image_final.png")

    # final_diff = torch.abs(modifier)
    # final_diff_img = tensor2img(final_diff)
    # final_diff_img.save("modifier_diff_final.png")
    # wandb.finish()
    # np.savetxt('adv_latent.txt', adv_latent.detach().cpu().numpy(), fmt='%f')
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

def process_images(source_img_path, target_img_path_1, target_img_path_2, target_img_path_3, target_img_path_4):
    source_image = cv2.imread(source_img_path)
    
    target_image_1 = cv2.imread(target_img_path_1)
    target_image_2 = cv2.imread(target_img_path_2)
    target_image_3 = cv2.imread(target_img_path_3)
    target_image_4 = cv2.imread(target_img_path_4)

    source_face = get_face_bbox(source_image)
    
    target_face_2 = get_face_bbox(target_image_2)
    target_face_3 = get_face_bbox(target_image_3)
    target_face_4 = get_face_bbox(target_image_4)
    target_face_1 = get_face_bbox(target_image_1)

    if source_face is None:
        print("One of the images does not contain a detectable face.")
        return None
    source_bbox = source_face.bbox.astype(int).tolist()
    
    print(":::::bbox::::::", source_bbox)

    cropped_source_face = crop_face_bbox(source_image, source_face)
    
    cropped_target_face_1 = crop_face_bbox(target_image_1, target_face_1)    
    cropped_source_face_2 = crop_face_bbox(target_image_2, target_face_2)
    cropped_source_face_3 = crop_face_bbox(target_image_3, target_face_3)
    cropped_source_face_4 = crop_face_bbox(target_image_4, target_face_4)

    cv2.imwrite(os.path.join(os.path.dirname(__file__), './source_face.png'), cropped_source_face)
    
    cv2.imwrite(os.path.join(os.path.dirname(__file__), './target_face_1.png'), cropped_target_face_1)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), './target_face_2.png'), cropped_source_face_2)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), './target_face_3.png'), cropped_source_face_3)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), './target_face_4.png'), cropped_source_face_4)
    print("Cropped faces saved.")

    return source_bbox

def replace_image(original_image: np.ndarray, cropped_image, coords):
    # 좌표 가져오기
    x1, y1, x2, y2 = coords
    target_height = y2 -y1
    target_width = x2 - x1

    # 크롭된 이미지를 coords 크기에 맞게 변형
    if isinstance(cropped_image, Image.Image):
        cropped_image = np.array(cropped_image)
    # 채널 순서 변환 (RGB -> BGR)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    resized_cropped_image = cv2.resize(cropped_image, (target_width, target_height))

    # 원본 이미지에 다시 삽입
    original_image[y1:y2, x1:x2] = resized_cropped_image
    cv2.imwrite("output\\resized_image.png", resized_cropped_image)

    return original_image

def replace_blend_image(original_image: np.ndarray, cropped_image, coords):
    # 좌표 가져오기
    x1, y1, x2, y2 = coords
    target_height = y2 -y1
    target_width = x2 - x1
    eighth_h = target_height // 8
    remaining_h = target_height - eighth_h * 2
    eighth_w = target_width // 8
    remaining_w = target_width - eighth_w * 2


    # 크롭된 이미지를 coords 크기에 맞게 변형
    if isinstance(cropped_image, Image.Image):
        cropped_image = np.array(cropped_image)
    # 채널 순서 변환 (RGB -> BGR)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    resized_cropped_image = cv2.resize(cropped_image, (target_width, target_height))

    # 데이터 타입 확인 및 변환 (float -> uint8)
    if resized_cropped_image.dtype != original_image.dtype:
        resized_cropped_image = resized_cropped_image.astype(original_image.dtype)
    

    # 알파 마스크 생성 (중심은 1, 모서리로 갈수록 0으로 감소)
    ## 가로세로 각 8등분해서 가운데 6칸씩은 무조건 100% 투명도 보장
    mask_y_q = np.linspace(0, 1, eighth_h)
    mask_y = np.concatenate((mask_y_q, np.ones(remaining_h), mask_y_q[::-1]), axis=0)[:, np.newaxis]

    mask_x_q = np.linspace(0, 1, eighth_w)
    mask_x = np.concatenate((mask_x_q, np.ones(remaining_w), mask_x_q[::-1]), axis=0)[np.newaxis, :]

    mask = np.minimum(mask_y, mask_x)

    mask = mask[..., np.newaxis]  # 채널 차원 추가

    # 블렌딩 (마스크가 3채널로 변환되어야 함)
    mask = np.repeat(mask, 3, axis=2)

    # 원본 이미지에 다시 삽입 (크기가 정확히 맞는지 확인)
    if resized_cropped_image.shape[:2] == (target_height, target_width):
        blended_image = (resized_cropped_image * mask + original_image[y1:y2, x1:x2] * (1 - mask)).astype(
            original_image.dtype)
        cv2.imwrite("output\\blended_image.png", blended_image)
        original_image[y1:y2, x1:x2] = blended_image
    else:
        print("Error: Resized cropped image dimensions do not match target dimensions.")

    # 원본 이미지에 다시 삽입
    # original_image[y1:y2, x1:x2] = resized_cropped_image
    return original_image

# main

device = "cuda"
eps = 0.1
t_size = 1000
onnx_model_path = netArc_checkpoint = os.path.join(os.path.dirname(__file__), '../model/w600k_r50.onnx')
onnx_model = onnx.load(onnx_model_path)
det_size=(320, 320)
face_analyser = get_face_analyser(det_size)

torch_model = convert(onnx_model)
torch_model = torch_model.to(device)
torch_model.eval()

netArc_checkpoint = os.path.join(os.path.dirname(__file__), '../model/arcface_checkpoint.tar')
netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
netArc = netArc_checkpoint
netArc = netArc.to(device)
netArc.eval()

src_image_path = os.path.join(os.path.dirname(__file__), "image/jw3.png")
target_image_path_1 = os.path.join(os.path.dirname(__file__), "image/jw1.png")
target_image_path_2 = os.path.join(os.path.dirname(__file__), "image/jw2.png")
target_image_path_3 = os.path.join(os.path.dirname(__file__), "image/jw4.png")
target_image_path_4 = os.path.join(os.path.dirname(__file__), "image/jw5.png")

src_bbox = process_images(src_image_path, target_image_path_1, target_image_path_2, target_image_path_3, target_image_path_4)
w_pool = [0.1, 0.5, 1.0]
# for i, j, k in permutations(w_pool, 3):
#     noised_img = generate_one(os.path.join(os.path.dirname(__file__), 'source_face.png'), 
#                             os.path.join(os.path.dirname(__file__), 'target_face_1.png'),
#                             os.path.join(os.path.dirname(__file__), 'target_face_2.png'),
#                             os.path.join(os.path.dirname(__file__), 'target_face_3.png'),
#                             os.path.join(os.path.dirname(__file__), 'target_face_4.png'),
#                             i, j, k) 

noised_img = generate_one(os.path.join(os.path.dirname(__file__), 'source_face.png'), 
                        os.path.join(os.path.dirname(__file__), 'target_face_1.png'),
                        os.path.join(os.path.dirname(__file__), 'target_face_2.png'),
                        os.path.join(os.path.dirname(__file__), 'target_face_3.png'),
                        os.path.join(os.path.dirname(__file__), 'target_face_4.png'),
                        0.3, 0.3, 1.0) 

original_img = cv2.imread(src_image_path)
final_image = replace_blend_image(original_img, noised_img, src_bbox)

# cv2.imwrite(os.path.join(os.path.dirname(__file__), "final_image.png"), final_image)

image_dir = "/home/user/bob/fakeguard/analysis_result/greed"
cv2.imwrite(os.path.join(image_dir, ("eps_"+str(round(eps,2))+".png")), final_image)

print("Image processing completed!")
