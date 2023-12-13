# apt-get -y install libgl1-mesa-glx
# apt-get install libglib2.0-0
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
import torch
import random
import torch.nn.functional as F
import re
import math
from diffusers import DPMSolverMultistepScheduler
from models.stablediffusion import StableDiffusionPipeline
from matplotlib import pyplot as plt
from tqdm import tqdm
data_dir = "/home/pr07/jh_park/data/karpathy_test_text.json"
import json
with open(data_dir, 'r') as file:
    data = json.load(file)
from utils import *

import pytorch_fid_wrapper as pfw
from PIL import Image
from tqdm import tqdm
import json
with open(data_dir, 'r') as file:
    data = json.load(file)

def auto_device(obj = torch.device('cpu')):
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def set_seed(seed: int, cuda_num) -> torch.Generator:
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device= torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"))
    gen.manual_seed(seed)
    return gen

def normal_auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


import trace

# Hook Setting
gradients = dict()
activations = dict()

def backward_hook(module, grad_input, grad_output):
    # print("Backwarding!!")
    gradients['value'] = grad_output[0]

def forward_hook( module, input, output):
    activations['value'] = output


unet_config = {
    "up" : [
        {"resnets":[1,2,3], "upsamplers" : [1]},
        {"attentions": [1,2,3], "resnets":[1,2,3], "upsamplers" : [1]},
        {"attentions": [1,2,3], "resnets":[1,2,3], "upsamplers" : [1]},
        {"attentions": [1,2,3], "resnets":[1,2,3]},
            ],
    "down" : [
        {"attentions": [1,2], "resnets":[1,2], "downsamplers" : [1]},
        {"attentions": [1,2], "resnets":[1,2], "downsamplers" : [1]},
        {"attentions": [1,2], "resnets":[1,2], "downsamplers" : [1]},
        {"resnets":[1,2]},
        ],
    "mid" : [{"resnets":[1,2]}]
    }

def diffusion_gradcam(pipeline, prompt_txt, seed, num_step, step_vis, layer_message, time_steps, gradients, activations, cuda_num):
    
    out, logit, all_images, auc_dict = pipeline(
        prompt_txt,
        num_inference_steps=num_step,
        generator=set_seed(seed, cuda_num),
        get_images_for_all_inference_steps = True,
        output_type = None,
        step_visualization_num = step_vis,
        time_steps = time_steps,
        visualization_mode = {'mode':"cam", 'mask':None , 'layer_vis' : False, 'auc_options' : {'auc':True, "auc_type": None}}
        )
    # print(logit.samplle.sum())
    logit.sample.sum().backward()

    guidance_scale = 7.5
    
    grad_pred_uncond, grad_pred_text = gradients['value'].data.chunk(2)
    gradients_ = grad_pred_uncond + guidance_scale * (grad_pred_text - grad_pred_uncond)
    if "attentions" in layer_message:
        actv_pred_uncond, actv_pred_text = activations['value'].sample.chunk(2)
        activations_ = actv_pred_uncond + guidance_scale * (actv_pred_text - actv_pred_uncond)
    else:
        actv_pred_uncond, actv_pred_text = activations['value'].data.chunk(2)
        activations_ = actv_pred_uncond + guidance_scale * (actv_pred_text - actv_pred_uncond)

    b, k, u, v = activations_.size()
    # Mean of feature maps
    alpha = gradients_.view(b, k, -1).mean(2)
    weights = alpha.view(b, k, 1, 1)
    saliency_map = (weights * activations_).sum(1, keepdim=True)

    h = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    w = pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    saliency_map = F.relu(saliency_map)

    # Upsampling
    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    # normalization
    saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

    return out, all_images, saliency_map, auc_dict
    
def layer_setting(pipe, block, layer, block_num, layer_num):
    if block == "down":
        # In the case of diffusion
        if layer == "attentions":
            for_hook_handle = pipe.unet.down_blocks[block_num].attentions[layer_num-1].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.down_blocks[block_num].attentions[layer_num-1].register_backward_hook(backward_hook)
        elif layer == "resnets":
            for_hook_handle =pipe.unet.down_blocks[block_num].resnets[layer_num-1].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.down_blocks[block_num].resnets[layer_num-1].register_backward_hook(backward_hook)
        elif layer == "downsamplers":
            for_hook_handle =pipe.unet.down_blocks[block_num].downsamplers[0].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.down_blocks[block_num].downsamplers[0].register_backward_hook(backward_hook)
        else:
            raise ValueError("layer should be 'attention' or 'resnet' or 'downsampler'")

    elif block == "up":
        if layer == "attentions":
            for_hook_handle =pipe.unet.up_blocks[block_num].attentions[layer_num-1].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.up_blocks[block_num].attentions[layer_num-1].register_backward_hook(backward_hook)
        elif layer == "resnets":
            for_hook_handle =pipe.unet.up_blocks[block_num].resnets[layer_num-1].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.up_blocks[block_num].resnets[layer_num-1].register_backward_hook(backward_hook)
        elif layer == "upsamplers":
            for_hook_handle =pipe.unet.up_blocks[block_num].upsamplers[0].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.up_blocks[block_num].upsamplers[0].register_backward_hook(backward_hook)
        else:
            raise ValueError("layer should be 'attention' or 'resnet' or 'upsamplers'")
        
    elif block == "mid":
        # one layer
        if layer == "attentions":
            for_hook_handle =pipe.unet.mid_block.attentions[0].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.mid_block.attentions[0].register_full_backward_hook(backward_hook)
        # two layers
        elif layer == "resnets":
            for_hook_handle =pipe.unet.mid_block.resnets[layer_num-1].register_forward_hook(forward_hook)
            back_hook_handle = pipe.unet.mid_block.resnets[layer_num-1].register_full_backward_hook(backward_hook)
    else:
        raise ValueError("block should be 'down' or 'up' or 'mid_block'")
        
    layer_msg = f" {block}Block-{block_num}_{layer}-{layer_num}"
    return layer_msg, for_hook_handle, back_hook_handle

def time_step_sampling(time_stage, num_step, lamda):
    if lamda == 0:
        return None
    alpha = math.e ** (math.log(1000)/(num_step+lamda))
    if time_stage == "early":
        time_steps = [1000-int(alpha ** (i+1+lamda))for i in range(num_step)]
        time_steps.insert(0, 999)
        time_steps.pop(-1)
        time_steps = np.array(time_steps)
        time_steps = torch.from_numpy(time_steps).to(device)
    elif time_stage == "latter":
        time_steps = [int(alpha ** (i+1+lamda))for i in range(num_step)]
        time_steps.sort(reverse=True)
        if time_steps[0] == 1000:
            time_steps[0] = 999
        time_steps = np.array(time_steps)
        time_steps = torch.from_numpy(time_steps).to(device)
    elif time_stage == "uniform":
        time_steps = None
    else:
        raise ValueError
    return time_steps
def decode_latents(pipe, latents):
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    torch.cuda.empty_cache()
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float()
    return image


def auc_run(pipe, mode, saliency_weights, auc_dict, stride = None, save_to=None, visualize_process=False, visualize_last=False):

    saliency_map = saliency_weights
    input_latent = auc_dict["input_latent"]
    target_pred = auc_dict["target_latents"]
    encoder_hidden = torch.cat([pipe.text_embeddings.unsqueeze(0)] * 2)

    randomize = False


    t = auc_dict['t']
    if stride == None:
        stride = input_latent.shape[-1]
    n_steps = (input_latent.shape[-1] ** 2 + stride - 1) // stride

    # saliency_map = F.relu(explanation).unsqueeze(0).unsqueeze(0)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    # normalization
    saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

    assert mode in ['del', 'ins', 'noise']
    if mode == 'del':
        substrate_fn = torch.zeros_like
        # substrate_fn = torch.rand_like
        # substrate_fn = torch.from_numpy(np.random.uniform(0, 1, size=(input_latent.shape[-1], input_latent.shape[-1])))
    elif mode == 'ins':
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig).to(saliency_weights.device)
        substrate_fn = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen//2).to(saliency_weights.device)
    elif mode == 'noise':
        substrate_fn = torch.from_numpy(np.random.uniform(0, 1, size=input_latent.shape))


    if mode == 'del':
        title = 'Deletion game'
        ylabel = 'Pixels deleted'
        start = input_latent.clone()
        finish = substrate_fn(input_latent)
    elif mode == 'ins':
        title = 'Insertion game'
        ylabel = 'Pixels inserted'
        start = substrate_fn(input_latent)
        finish = input_latent.clone()
    elif mode == 'noise':
        title = 'Noising game'
        ylabel = 'Pixels noised'
        start = input_latent.clone()
        finish = substrate_fn


    scores = np.empty(n_steps + 1)

    pfw.set_config(batch_size=1, device = target_pred.device)
    h,w = target_pred.shape[2:]
    target_img = decode_latents(pipe, target_pred)
    target_img = target_img.permute(0,3,1,2)

    # Coordinates of pixels in order of decreasing saliency
    # Random saliency
    if randomize:
        saliency_map = torch.rand_like(saliency_map)
    explanation = saliency_map.clone().detach().cpu().numpy()
    explanation = F.interpolate(torch.from_numpy(explanation).float(), size=(64, 64)).numpy()
    salient_order = np.flip(np.argsort(explanation.reshape(-1, n_steps**2), axis=1), axis=-1)
    # salient_order = np.argsort(explanation.reshape(-1, n_steps**2), axis=1)
    for i in range(n_steps+1):
        input_latents = torch.cat([start.to(saliency_weights.device)] * 2)
        pred = pipe.unet(input_latents, t, encoder_hidden_states=encoder_hidden).sample
        torch.cuda.empty_cache()
        noise_pred_uncond, noise_pred_text = pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)
        pred_latents = pipe.scheduler.step(noise_pred_cfg, t, start.to(saliency_weights.device), **pipe.extra_step_kwargs).prev_sample
        pred_img = decode_latents(pipe, pred_latents)
        pred = pred_img.permute(0,3,1,2)
        score = pfw.fid(pred, target_img)
        scores[i] = score.mean()
        # Render image if verbose, if it's the last step or if save is required.
        if visualize_process:
            if i == 0:
                plt.figure(figsize=(10, 5))
                plt.subplot(331)
                plt.title("Target image")
                plt.imshow(target_img.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off")
            elif i == 1:
                plt.subplot(332)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off")
            elif i == 3:
                plt.subplot(333)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off")
            elif i == 8:
                plt.subplot(334)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off")
            elif i == 30:
                plt.subplot(335)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off") 
            elif i == 40:
                plt.subplot(336)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off") 
            elif i == 50:
                plt.subplot(337)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off") 
            elif i == 70:
                plt.subplot(338)
                plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
                plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach().numpy())
                plt.axis("off") 

        if i == n_steps:
            if visualize_last:
                plt.figure(figsize=(10, 5))
                # plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                    # plt.subplot(339)
                    
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                # plt.xlim(-0.1, 1.1)
                # plt.ylim(, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel("score")
                if save_to:
                    plt.savefig(save_to + f'/{mode}_fid.png')
                    plt.show()
                    plt.close()
                else:
                    plt.show()
                return scores

            else:
                return scores

        coords = salient_order[:, stride * i:stride * (i + 1)]
        start = start.cpu().numpy().reshape(1, 4, n_steps**2)

        start[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords]
        start = torch.from_numpy(start.reshape(1,4,n_steps,n_steps))
        torch.cuda.empty_cache()
        # start.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords]
            # start.cpu().numpy().reshape(-1, n_steps**2)[:, coords] = finish.cpu().numpy().reshape(-1, n_steps**2)[:, coords]
    



# Model loading...
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "CompVis/stable-diffusion-v1-4"

cuda_num = 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "{cuda_num}"  # Set the GPUs 2 and 3 to use
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(f"cuda:{cuda_num}")
net = torch.nn.DataParallel(pipe).to(device)
pipe = net.module
# pipe.unet.config.sample_size = 64

print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())



inference_steps = 20
visualization_step = 15
seed_num = 1
auc = 0.
# mode = 'del'
mode = 'ins'

for idx, batch in tqdm(enumerate(data[0:1000]), desc = f"AUC score mode: {mode} >>> "): 
# for batch in tqdm(data[500:1001]):
    torch.cuda.empty_cache()
    prompt = batch["targets"]
    saliency_weights = torch.zeros((1,1,512,512),  dtype = torch.float16).to(f"cuda:{cuda_num}")

    for block_name, block in unet_config.items():
            for block_num, layer_dict in enumerate(block):
                    for layer_name, layer_lst in layer_dict.items():
                            for layer_num in layer_lst:
                                    layer_msg, for_hook_handle, back_hook_handle = layer_setting(pipe, block_name, layer_name, block_num, layer_num)
                                    output, all_images, sal_map, auc_dict = diffusion_gradcam(
                                            pipeline = pipe, 
                                            prompt_txt = prompt, 
                                            seed =1, 
                                            num_step = inference_steps, 
                                            step_vis = visualization_step,
                                            layer_message = layer_msg,
                                            time_steps = None,
                                            gradients= gradients,
                                            activations=activations,
                                            cuda_num = cuda_num
                                            )
                                    saliency_weights += sal_map
                                    torch.cuda.empty_cache()
                                    for_hook_handle.remove()
                                    back_hook_handle.remove()

    auc_score = auc_run(pipe, mode = mode, saliency_weights= saliency_weights, auc_dict = auc_dict, stride = None, save_to=None, visualize_process=False, visualize_last=False)
    torch.cuda.empty_cache()
    auc += auc_score
    if idx % 10 == 0:
        auc_middle = auc/1000.
        auc_middle_min, auc_middle_max = auc_middle.min(), auc_middle.max()
        # normalization
        auc_middle_minmax = (auc_middle - auc_middle_min) / (auc_middle_max - auc_middle_min)
        print('{} : AUC: {}'.format(idx, normal_auc(auc_middle_minmax)))

auc = auc/1000.
auc_score_min, auc_score_max = auc.min(), auc.max()
# normalization
auc_minmax = (auc - auc_score_min) / (auc_score_max - auc_score_min)
print('AUC: {}'.format(normal_auc(auc_minmax)))