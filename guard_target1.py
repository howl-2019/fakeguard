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

# tensorboard 벡터 기록
# from torch.utils.tensorboard import SummaryWriter
# import pandas as pd

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

def generate_one(src_image_path, target_image_path, mask):
    if not os.path.exists(src_image_path):
        raise FileNotFoundError(f"Source image not found: {src_image_path}")
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    source_tensor = preprocess_image(src_image_path)
    target_tensor = preprocess_image(target_image_path)

    source_tensor = source_tensor.to(device)
    target_tensor = target_tensor.to(device)

    with torch.no_grad():
        target_auto = torch_model(target_tensor)
        target_gan = netArc(target_tensor)
    
    modifier = torch.zeros_like(source_tensor, requires_grad=True) # 색 있는 거 
    # modifier = torch.zeros_like(source_tensor[:, :1, :, :], requires_grad=True)
    optimizer = optim.Adam([modifier], lr=0.01)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    max_change = eps / epss #0.5  # scale from 0,1 to -1,1
    step_size = max_change

    for i in range(t_size):
        # actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
        
        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1) # adv, modifier connect
        # adv_tensor = torch.clamp(modifier * mask + source_tensor, -1, 1) # adv, modifier connect

        adv_tensor = adv_tensor.to(device)
        adv_auto = torch_model(adv_tensor)
        adv_gan = netArc(adv_tensor)
        
        uc_distance_auto = (adv_auto - target_auto).norm()
        dual_dis_auto = 0.5 * torch.abs(adv_auto - target_auto).sum() + 0.5 * torch.norm(adv_auto - target_auto)
        mse_loss_auto = torch.nn.functional.mse_loss(adv_auto, target_auto)
        cosine_loss_auto = 1 - torch.nn.functional.cosine_similarity(adv_auto, target_auto, dim=0).mean()
        
        uc_distance_gan = (adv_gan - target_gan).norm()
        dual_dis_gan = 0.5 * torch.abs(adv_gan - target_gan).sum() + 0.5 * torch.norm(adv_gan - target_gan) 
        mse_loss_gan = torch.nn.functional.mse_loss(adv_gan, target_gan)
        cosine_loss_gan = 1 - torch.nn.functional.cosine_similarity(adv_gan, target_gan, dim=0).mean()
        
        
        loss_auto = dual_dis_auto + mse_loss_auto
        loss_gan = dual_dis_gan + mse_loss_gan
        # loss = -(loss_auto + loss_gan)
        loss = (loss_auto + loss_gan)

        wandb.log({"loss": loss, "L2_distance_auto": uc_distance_auto, "cosine_auto": cosine_loss_auto, "mse_loss": mse_loss_auto , "L2_distance_gan": uc_distance_gan, "cosine_gan": cosine_loss_gan, "mse_gan": mse_loss_gan})
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(torch_model.parameters()) + list(netArc.parameters()), max_norm=1.0)
        
        optimizer.step()
        # scheduler.step()
        modifier.data.clamp_(-max_change, max_change)
        

        if i % 50 == 0:
            print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")

            # iter_img = tensor2img(modifier + source_tensor)
            # iter_img.save(f"modifier_image_iter_{i}.png")

            # diff = torch.abs(modifier)
            # diff_img = tensor2img(diff)
            # diff_img.save(f"modifier_diff_iter_{i}.png")

    final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

    final_img = tensor2img(final_adv_batch)
    final_img.save("final_image_sy8_dog.png")

    final_modifier_img = tensor2img(modifier + source_tensor)
    final_modifier_img.save("modifier_image_final_sy8_dog.png")

    # final_diff = torch.abs(modifier)
    # final_diff_img = tensor2img(final_diff)
    # final_diff_img.save("modifier_diff_final.png")
    wandb.finish()
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

def facial_feature_mask(face, mask_shape):

    landmarks = face.kps  # Facial landmarks (left_eye, right_eye, nose, mouth)
    x_min, y_min, x_max, y_max = face.bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    height, width = 112, 112

    normalized_landmarks = [
        ( (coord[0] - x_min) / bbox_width * 112,
        (coord[1] - y_min) / bbox_height * 112 )
        for coord in landmarks
    ]

    # Initialize an empty mask
    mask = torch.zeros((height, width))

    # Define regions for eyes, nose, and mouth
    # Eyes
    for eye in normalized_landmarks[:2]:
        eye_x, eye_y = eye
        mask[
            int(eye_y - 11):int(eye_y + 11),
            int(eye_x - 17):int(eye_x + 17)
        ] = 1

    # Nose
    nose_x, nose_y = normalized_landmarks[2]
    mask[
        int(nose_y - 17):int(nose_y + 17),
        int(nose_x - 11):int(nose_x + 11)
    ] = 1

    # Mouth
    left_mouth_x, left_mouth_y = normalized_landmarks[3]
    right_mouth_x, right_mouth_y = normalized_landmarks[4]
    mouth_center_y = (left_mouth_y + right_mouth_y) / 2
    mask[
        int(mouth_center_y - 11):int(mouth_center_y + 17),
        int(left_mouth_x - 11):int(right_mouth_x + 11)
    ] = 1

    assert mask.max() <= 1 and mask.min() >= 0, "Mask should contain only 0 and 1."

    # mask = mask.numpy()

    # if len(mask.shape) == 2:  # If mask is single-channel
    #     mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand to 3 channels

    mask = mask.to(device)
    return mask

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

    cv2.imwrite(os.path.join(os.path.dirname(__file__), './source_face_sy8.png'), cropped_source_face)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), './target_face_dog.png'), cropped_target_face)
    print("Cropped faces saved.")

    # 마스크 생성
    mask_shape = cropped_source_face.shape[:2]
    src_mask = facial_feature_mask(source_face, mask_shape)

    return source_bbox, src_mask

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
    cv2.imwrite("output/resized_image_pm10.png", resized_cropped_image)

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
        cv2.imwrite("output/blended_image_pm10.png", blended_image)
        original_image[y1:y2, x1:x2] = blended_image
    else:
        print("Error: Resized cropped image dimensions do not match target dimensions.")

    # 원본 이미지에 다시 삽입
    # original_image[y1:y2, x1:x2] = resized_cropped_image
    return original_image

# main

device = "cuda"
eps = 1
epss = 0.5
t_size = 300
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


for param in torch_model.parameters():
    print("torch_model is on device:", param.device)
    break  # 첫 번째 파라미터만 확인

# netArc의 첫 번째 파라미터의 장치를 확인
for param in netArc.parameters():
    print("netArc is on device:", param.device)
    break  # 첫 번째 파라미터만 확인

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Current GPU device: {current_device}")
    print(f"GPU device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available.")

src_image_path = os.path.join(os.path.dirname(__file__), "image/go1.jpg")
target_image_path = os.path.join(os.path.dirname(__file__), "image/go3.jpg")

src_bbox, src_mask = process_images(src_image_path, target_image_path)

# 이목구비만 남길 마스크 생성
# src_mask = facial_feature_mask(get_face_bbox(cv2.imread(src_image_path)))

noised_img = generate_one(os.path.join(os.path.dirname(__file__), 'source_face_sy8.png'), 
                          os.path.join(os.path.dirname(__file__), 'target_face_dog.png'),
                          src_mask)

original_img = cv2.imread(src_image_path)
final_image = replace_blend_image(original_img, noised_img, src_bbox)

cv2.imwrite(os.path.join(os.path.dirname(__file__), "final_image_sy8_dog.png"), final_image)
print("Image processing completed!")



