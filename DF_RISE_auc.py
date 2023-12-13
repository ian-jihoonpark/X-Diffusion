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

def auto_device(obj = torch.device('cpu')):
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)
    return gen

def normal_auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

# Model loading...
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "CompVis/stable-diffusion-v1-4"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPUs 2 and 3 to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
net = torch.nn.DataParallel(pipe).to(device)
pipe = net.module
# pipe.unet.config.sample_size = 64

print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())



inference_steps = 20
visualization_step = 15
num_masks = 500
seed_num = 1
auc = 0.
for batch in tqdm(data[0:1000]):
# for batch in tqdm(data[500:1001]):
    torch.cuda.empty_cache()
    prompt = batch["targets"]

    all_images, rise, auc_score, output = pipe(
        prompt,
        num_inference_steps=inference_steps,
        generator=set_seed(seed_num),
        get_images_for_all_inference_steps = True,
        output_type =None,
        step_visualization_num = visualization_step,
        visualization_mode = {'mode' : "rise", 'mask' : "gaussian", 'layer_vis' : False, 'auc_option':{'auc':True, 'auc_step':10}},
        rise_num_steps = num_masks,
        similarity_args = {"sim_func": "ssim", "ssim_mode" : "structure"},
        time_steps = None,
        )
    auc+=auc_score
auc = auc/1000.
auc_score_min, auc_score_max = auc.min(), auc.max()
# normalization
auc_minmax = (auc - auc_score_min) / (auc_score_max - auc_score_min)
print('AUC: {}'.format(normal_auc(auc_minmax)))