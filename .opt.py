import numpy as np
from PIL import Image
from facedetector import process_single_tensor_image
import os
from diffusers import StableDiffusionPipeline
import torch
import torch.utils.data
from einops import rearrange
from torchvision import transforms

class PoisonGeneration(object):
    def __init__(self, target_concept, device, eps=0.05):
        self.eps = eps
        self.target_concept = target_concept
        self.device = device
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def generate_target(self, prompts):
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = self.full_sd_model(prompts, guidance_scale=7.5, num_inference_steps=50, height=512, width=512).images
        target_imgs[0].save("target.png")
        return target_imgs[0]

    def get_latent(self, tensor):
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features

    def generate_one(self, src_image, target_image):

        source_tensor = img2tensor(src_image).to(self.device)

        target_tensor = img2tensor(target_image).to(self.device)

        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()

        target_latent = process_single_tensor_image(target_tensor)

#modifier = torch.clone(source_tensor) * 0.0
        modifier = torch.zeros_like(source_tensor, requires_grad=True)
        t_size = 500
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
#modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = process_single_tensor_image(adv_tensor)
            
            if isinstance(adv_latent, np.ndarray):
                adv_latent_tensor = torch.from_numpy(adv_latent).to(self.device)
            else:
                adv_latent_tensor = adv_latent
            
            if isinstance(target_latent, np.ndarray):
                target_latent_tensor = torch.from_numpy(target_latent).to(self.device)
            else:
                target_latent_tensor = target_latent
            
            adv_latent_tensor.requires_grad_(True)
            target_latent_tensor.requires_grad_(True)
            modifier.requires_grad_(True)
            print(adv_latent_tensor, target_latent_tensor)
            print(adv_latent_tensor.shape, target_latent_tensor.shape)
            print(modifier.requires_grad)
            loss = (adv_latent_tensor - target_latent_tensor).norm()
            
            tot_loss = loss.sum()
#grad = torch.autograd.grad(tot_loss, modifier)[0]
            tot_loss.requires_grad_(True)
            grad = torch.autograd.grad(tot_loss, modifier, retain_graph=True, allow_unused=True)[0]
            print(grad)
            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
#modifier = modifier.detach()

            if i % 50 == 0:
                print("# Iter: {}\tLoss: {:.3f}".format(i, loss.mean().item()))
								# 섭동이 추가된 이미지를 png 파일로 저장하는 코드
                iter_img = tensor2img(modifier + source_tensor)
                iter_img.save(f"modifier_image_iter_{i}.png")
								# 섭동만 따로 png로 저장하는 코드
                diff = torch.abs(modifier)
                diff_img = tensor2img(diff)
                diff_img.save(f"modifier_diff_iter_{i}.png")

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
				# 최종 결과물을 png로 저장하는 코드
        final_img = tensor2img(final_adv_batch)
        final_modifier_img = tensor2img(modifier + source_tensor)
        final_modifier_img.save("modifier_image_final.png")
				# 마지막 섭동을 png로 저장하는 코드
        final_diff = torch.abs(modifier)
        final_diff_img = tensor2img(final_diff)
        final_diff_img.save("modifier_diff_final.png")
        return final_img



def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img
