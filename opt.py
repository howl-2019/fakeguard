import os
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from torchvision import transforms
from facedetector import FaceSwap

class PoisonGeneration(object):
    def __init__(self, device, eps=0.05):
        self.eps = eps
        self.device = device
        self.transform = self.resizer()
        self.face_swapper = FaceSwap(model_path="../model/w600k_r50.onnx")
    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms


    def generate_one(self, src_image_path, target_image_path):
        if not os.path.exists(src_image_path):
            raise FileNotFoundError(f"Source image not found: {src_image_path}")
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        source_tensor = img2tensor(self.transform(Image.open(src_image_path))).to(self.device)
        target_tensor = img2tensor(self.transform(Image.open(target_image_path))).to(self.device)

        source_tensor.requires_grad_()
        target_tensor.requires_grad_()

        # target_tensor = target_tensor.half()
        # source_tensor = source_tensor.half()

        target_latent = self.face_swapper.extract_face_embedding(target_tensor)

        modifier = torch.zeros_like(source_tensor, requires_grad=True)

        t_size = 1000
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1) # adv, modifier connect
            adv_latent = self.face_swapper.extract_face_embedding(adv_tensor)
            
            with torch.enable_grad():
                #loss = (adv_latent - target_latent).norm()
                loss = 0.5 * torch.abs(adv_latent - target_latent).sum() + 0.5 * torch.norm(adv_latent - target_latent)
                #loss = torch.nn.functional.kl_div(adv_latent.log_softmax(dim=-1), target_latent, reduction='batchmean')            
                #loss = torch.nn.functional.mse_loss(adv_latent, target_latent)
                #loss = 1 - torch.nn.functional.cosine_similarity(adv_latent, target_latent, dim=0).mean()
                tot_loss = loss.sum()

            if not tot_loss.requires_grad:
                tot_loss = tot_loss.requires_grad_()
            
            # tot_loss.backward()
            # print(modifier.grad)

            grad = torch.autograd.grad(tot_loss, modifier, allow_unused=False)[0]
            if grad is None:
                print("skip")
                continue
            # if modifier.grad is None:
            #     print("skip")
            #     continue
            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)

            if i % 50 == 0:
                print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")

                # Save images of the current perturbation
                # iter_img = tensor2img(modifier + source_tensor)
                # iter_img.save(f"modifier_image_iter_{i}.png")

                # # Save the difference
                # diff = torch.abs(modifier)
                # diff_img = tensor2img(diff)
                # diff_img.save(f"modifier_diff_iter_{i}.png")

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)

        # Save the final results
        final_img = tensor2img(final_adv_batch)
        final_img.save("final_image.png")

        final_modifier_img = tensor2img(modifier + source_tensor)
        final_modifier_img.save("modifier_image_final.png")

        final_diff = torch.abs(modifier)
        final_diff_img = tensor2img(final_diff)
        final_diff_img.save("modifier_diff_final.png")

        return [final_img, final_modifier_img, final_diff_img]


def img2tensor(cur_img):
    #cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
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

