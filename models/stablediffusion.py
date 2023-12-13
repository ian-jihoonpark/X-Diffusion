# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
import torch
from torch.nn import functional as F

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from models.clip_encoder import CLIPTextModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from models.unet_model import UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import math
import cv2
import pytorch_fid_wrapper as pfw
from tqdm import tqdm
from ssim import SSIM
from utils import *
import os
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from functools import partial
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from models.lime_model import LimeBase, ImageExplanation
import copy




logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# @dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            self.uncond_embeddings = uncond_embeddings
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            self.text_embeddings = text_embeddings[0]
        return text_embeddings

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        visualization_mode = {'mode':"rise", 'mask':'gaussian' , 'layer_vis' : False, 'auc_option' : {}}, # or "cam"
        rise_num_steps = 10,
        get_images_for_all_inference_steps = False,
        step_visualization_num = -1,
        time_steps = None,
        similarity_args = {},
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self.guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # with torch.no_grad():
        self.text_encoder.requires_grad_(True)
        text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if time_steps is None:
            timesteps = self.scheduler.timesteps
        else:
            timesteps = time_steps
            self.scheduler.timesteps = time_steps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        # (1, 4, 96, 96)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        self.output = []
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        all_generated_images = [] if get_images_for_all_inference_steps else None
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if visualization_mode['mode'] == "rise":
            self.output = []
            self.unet.down_blocks[2].resnets[1].register_forward_hook(self.forward_hook)
            self.masking_mode = visualization_mode['mask']
            self.gradients = dict()
            self.activations = dict()
            self.actv_map = None
        self.res =[]
        for i, t in enumerate(timesteps):
            input_latents = latents.clone().detach()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


            # predict the noise residual
            if i+1 == step_visualization_num:
                output = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = output.sample.clone().detach()
                self.output = output
            else:
                with torch.no_grad():
                    output = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                    noise_pred = output.sample.clone().detach()

            # decode latents at each steps
            if get_images_for_all_inference_steps:
                with torch.no_grad():
                    image = self.decode_latents(
                        latents=latents
                    )
                    all_generated_images.append(image)

            # perform guidance
            if do_classifier_free_guidance:

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_cfg, t, latents, **self.extra_step_kwargs).prev_sample
            else:
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample
                
            # Saliency map visualization
            if i+1 == step_visualization_num:

                if visualization_mode['mode'] == "rise":
                        
                    if visualization_mode["mask"] == "actv":
                        self.heatmap, self.auc = self.generate_saliency_map(
                            rise_num_steps = rise_num_steps, 
                            target_latents = latents, 
                            input_latent= input_latents, 
                            t = t, 
                            hidden_states = text_embeddings, 
                            prob_thresh=0.5,
                            activation_map= self.activations["value"],
                            layer_vis = visualization_mode["layer_vis"],
                            get_auc_score =visualization_mode["auc_options"]["auc"] is True,
                            **similarity_args
                            )
                    else:
                        if not visualization_mode["auc_options"]["auc"] is True:
                            self.heatmap, self.auc = self.generate_saliency_map(
                                rise_num_steps = rise_num_steps, 
                                target_latents = latents, 
                                input_latent= input_latents, 
                                t = t, 
                                hidden_states = text_embeddings, 
                                prob_thresh=0.5,
                                activation_map= None,
                                layer_vis = visualization_mode["layer_vis"],
                                get_auc_score = True,
                                **similarity_args
                                )
                # Lime
                elif visualization_mode['mode'] == "lime":
                    self.heatmap,self.auc_score = self.lime_visualization(
                        target_latents = latents, 
                        input_latent= input_latents, 
                        t = t, 
                        hidden_states = text_embeddings, 
                        activation_map= None,
                        layer_vis = visualization_mode["layer_vis"],
                        get_auc_score = True,
                        )
                elif visualization_mode['mode'] == "fid":
                    self.heatmap,self.auc_score = self.generate_saliency_map_fid(
                        target_latents = latents, 
                        input_latent= input_latents, 
                        t = t, 
                        hidden_states = text_embeddings, 
                        activation_map= None,
                        layer_vis = False,
                        get_auc_score = True,
                        sim_func = visualization_mode['mode']
                        )
                elif visualization_mode['mode'] == "rel_map":
                    with torch.no_grad():
                        image = self.decode_latents(latents)
                    image = self.transform_images_to_pil_format([image])[0]
                    return image, self.output
                elif visualization_mode['mode'] == "cam":
                        # res ,_ = self.generate_saliency_map(
                        # rise_num_steps = rise_num_steps, 
                        # target_latents = latents, 
                        # input_latent= input_latents, 
                        # t = t, 
                        # hidden_states = text_embeddings,
                        # prob_thresh=0.5,
                        # activation_map= None,
                        # layer_vis = False,
                        # ssim_mode = None,
                        # get_auc_score = False,
                        # auc_mode= visualization_mode['auc_options']['auc_type'],
                        # )
                        # self.res =res
                        pass
        # 8. Post-processing
        with torch.no_grad():
            image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        if all_generated_images:
            all_generated_images.append(image)

        if output_type == "pil":
            if all_generated_images:
                all_generated_images = self.transform_images_to_pil_format(all_generated_images)
                image = all_generated_images[-1]
            else:
                image = self.transform_images_to_pil_format([image])[0]
        if not return_dict:
            return (image, has_nsfw_concept)
        if visualization_mode['mode'] == "rise":
            return all_generated_images, self.heatmap, self.auc, self.output
        elif visualization_mode['mode'] == "lime":
            return all_generated_images, self.heatmap, self.output, self.auc_score
        else:
            if visualization_mode["auc_options"]["auc"] is True:
                outputs_dict = {
                    "input_latent" :  torch.cat([input_latents] * 2).to("cuda"),
                    "target_latents": latents,
                    "t": t ,
                    "input_latent": input_latents,
                    "random" : False,
                    "verbose":0,
                    "save_to":"outputs/auc",
                    'res': self.res
                    }

                return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), self.output, all_generated_images, outputs_dict

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), self.output, all_generated_images, 


    def transform_images_to_pil_format(self,all_generated_images: List[torch.Tensor]):
        pil_images = []
        for im in all_generated_images:
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            im = self.numpy_to_pil(im)
            pil_images.append(im)
        return pil_images

    def grid_masking_latents(self, latents, grid_size, prob_thresh):
        image_w, image_h = latents.shape[2:]
        grid_w, grid_h = grid_size
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

        mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < prob_thresh).astype(np.float32)
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
        ltnt = latents.permute(2,3,1,0).squeeze(-1)
        masked_latent = (ltnt.to(torch.float32) * torch.from_numpy(np.dstack([mask] * 4)).to(ltnt.device))
        masked_latent = masked_latent.permute(2,0,1).unsqueeze(0).to(torch.float16)
        mask = torch.from_numpy(mask).to(ltnt.device)
        return mask, masked_latent
    
    def gau_masking_latents(self, latents, prob_thresh):
        channel, image_w, image_h = latents.shape[1:]
        mask = (np.random.uniform(0, 1, size=(image_w, image_h)) < prob_thresh).astype(np.float32)
        ltnt = latents.permute(2,3,1,0).squeeze(-1)
        masked_latent = (ltnt.to(torch.float32) * torch.from_numpy(np.dstack([mask] * channel)).to(ltnt.device))
        masked_latent = masked_latent.permute(2,0,1).unsqueeze(0).to(torch.float16)
        mask = torch.from_numpy(mask).to(ltnt.device)
        return mask, masked_latent

    def actv_masking_latents(self, latents, prob_thresh, activ):
        image_w, image_h = latents.shape[2:]

        actv_pred_uncond, actv_pred_text = activ.data.chunk(2)
        activations_ = actv_pred_uncond + 0.7 * (actv_pred_text - actv_pred_uncond)

        activation_map = activations_.sum(1).unsqueeze(0)
        activation_map = F.interpolate(activation_map, size=(image_w, image_h), mode='bilinear', align_corners=False)

        mean = activation_map.mean()
        dice = np.random.randint(0,4)
        if dice == 0:
            actv_mask = torch.where(activation_map < mean, 1, 0).squeeze(0).squeeze(0).cpu().detach().numpy()
            mask = np.random.uniform(0, 1, size=(image_w, image_h))
            mask = mask * actv_mask
            mask = (mask > 0.1).astype(np.float32)

        elif dice == 1:
            actv_mask = torch.where(activation_map > mean, 1, 0).squeeze(0).squeeze(0).cpu().detach().numpy()
            mask = np.random.uniform(0, 1, size=(image_w, image_h))
            mask = mask * actv_mask
            mask = (mask > 0.3).astype(np.float32)

        else:
            mask = (np.random.uniform(0, 1, size=(image_w, image_h)) < prob_thresh).astype(np.float16)

        ltnt = latents.permute(2,3,1,0).squeeze(-1)
        masked_latent = (ltnt.to(torch.float32) * torch.from_numpy(np.dstack([mask] * 4)).to(ltnt.device))
        masked_latent = masked_latent.permute(2,0,1).unsqueeze(0).to(torch.float16)
        mask = torch.from_numpy(mask).to(ltnt.device)
        return mask, masked_latent


    @torch.no_grad()
    def generate_saliency_map_fid(
            self, 
            rise_num_steps, 
            target_latents, 
            input_latents,
            t, 
            hidden_states, 
            grid_size, 
            prob_thresh,
            get_auc_score = True,
            sim_func = "fid"
            ):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        pfw.set_config(batch_size=1, device = target_latents.device)
        h,w = target_latents.shape[2:]
        res = torch.zeros((h, w), dtype=torch.float32).to(target_latents.device)
        target_img = self.decode_latents(target_latents)
        target_img = torch.from_numpy(target_img).permute(0,3,1,2)
        # for step in tqdm(range(rise_num_steps), desc = "Rise iteration..."):
        for step in range(rise_num_steps):
            mask, masked_latents = self.masking_latents(input_latents, grid_size=grid_size, prob_thresh=prob_thresh)
            masked_latents_input = torch.cat([masked_latents] * 2).to("cuda")
            maksed_pred = self.unet(masked_latents_input, t, encoder_hidden_states=hidden_states)
            # Classifier free guidance
            masked_noise_pred_uncond, masked_noise_pred_text = maksed_pred.sample.chunk(2)
            masked_noise_pred_cfg = masked_noise_pred_uncond + self.guidance_scale * (masked_noise_pred_text - masked_noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1

            masked_latents = self.scheduler.step(masked_noise_pred_cfg, t, masked_latents, **self.extra_step_kwargs).prev_sample
            pred_img = self.decode_latents(masked_latents)
            pred = torch.from_numpy(pred_img).permute(0,3,1,2)

            plt.figure('Diffusion RISE', figsize=(10,4))
            plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            plt.axis("off")
            if sim_func == "fid":
                score = pfw.fid(pred, target_img)
            else:
                score = cos(target_img, pred).squeeze(0)
            res += mask * score

        if get_auc_score:
            mode = "ins"
            auc_score = self.auc_run(mode, input_latents, target_latents, res, t , hidden_states, random = False, verbose=0, save_to="outputs/auc")
        else:
            auc_score = None
        
        return res, pred_img

    @torch.no_grad()
    def generate_saliency_map(
            self, 
            rise_num_steps, 
            target_latents, 
            input_latent,
            t, 
            hidden_states, 
            prob_thresh,
            activation_map = None,
            sim_func = None,
            layer_vis = False,
            ssim_mode = None,
            get_auc_score = False,
            auc_mode ='ins'
            ):
        
        if layer_vis:
            self.unet.down_blocks[2].resnets[1].register_forward_pre_hook(self.preforward_hook)
            self.activations = dict()
            input_latents = torch.cat([input_latent] * 2).to("cuda")
            target_latents = target_latents

        h,w = target_latents.shape[2:]

        res = torch.zeros((h, w), dtype=torch.float32).to(target_latents.device)

        if sim_func == 'cos':
            score_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif sim_func == 'ssim':
            score_func = SSIM(15, reduction='none', mode = ssim_mode)

        # for step in tqdm(range(rise_num_steps), desc = "Rise iteration..."):
        for step in range(rise_num_steps):
            if not layer_vis:
                # Activation map version
                if activation_map is not None:
                    self.mask, masked_latents = self.actv_masking_latents(input_latent, prob_thresh=prob_thresh, activ = activation_map)
                else:
                # Gaussian random masking version
                    self.mask, masked_latents = self.gau_masking_latents(input_latent, prob_thresh=prob_thresh)

                masked_latents_input = torch.cat([masked_latents] * 2).to("cuda")
                masked_pred = self.unet(masked_latents_input, t, encoder_hidden_states=hidden_states).sample
                mask = self.mask
            else:
                if activation_map is not None:
                    self.actv_map = activation_map

                masked_pred = self.unet(input_latents, t, encoder_hidden_states=hidden_states).sample
                mask = self.mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            # Classifier free guidance
            masked_noise_pred_uncond, masked_noise_pred_text = masked_pred.chunk(2)
            masked_noise_pred_cfg = masked_noise_pred_uncond + self.guidance_scale * (masked_noise_pred_text - masked_noise_pred_uncond)
            
            # #  mask generation
            # masked_latents = self.scheduler.step(masked_noise_pred_cfg, t, masked_latents, **self.extra_step_kwargs).prev_sample
            # pred_img = self.decode_latents(masked_latents)
            # pred = torch.from_numpy(pred_img).permute(0,3,1,2)
            # if step <20:
            #     plt.figure('Mask', figsize=(10,4))
            #     plt.imshow(pred.squeeze(0).permute(1, 2, 0).cpu().numpy())
            #     plt.axis("off")
            #     plt.savefig(os.path.join(f"outputs/mask3/mask_{step}.png"))
            #     # plt.show()
                
            # ######### 
            auc_score =None
            score = score_func(target_latents, masked_noise_pred_cfg).squeeze(0)
            res += mask * score
        if get_auc_score:
            auc_score = self.auc_run(auc_mode, input_latent, target_latents, res, t , hidden_states, random = False, verbose=0, save_to="outputs/auc")
        else:
            auc_score = None
        return res , auc_score


    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output[0]
        pass

    def preforward_hook(self, module, input):
        noise_pred_uncond, noise_pred_text = input[0].chunk(2)
        target_latent = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.masking_mode == "actv":
            self.mask, masked_latents = self.actv_masking_latents(target_latent, prob_thresh=0.5, activ = self.actv_map)
        else:
            self.mask, masked_latents = self.gau_masking_latents(target_latent, prob_thresh=0.5)
        return (masked_latents, input[1])
    @torch.no_grad()
    def auc_run(self, mode, input_latent, target_pred, explanation, t, encoder_hidden, random = False, verbose=1, stride = None, save_to=None):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        if stride == None:
            stride = input_latent.shape[-1]
        n_steps = (input_latent.shape[-1] ** 2 + stride - 1) // stride

        saliency_map = F.relu(explanation).unsqueeze(0).unsqueeze(0)
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
            kern = gkern(klen, ksig).cuda()
            substrate_fn = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen//2).cuda()
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
        target_img = self.decode_latents(target_pred)
        target_img = torch.from_numpy(target_img).permute(0,3,1,2)

        # Coordinates of pixels in order of decreasing saliency
        # Random saliency
        if random:
            saliency_map = torch.rand_like(saliency_map)
        explanation = saliency_map.clone().detach().cpu().numpy()
        salient_order = np.flip(np.argsort(explanation.reshape(-1, n_steps**2), axis=1), axis=-1)
        # salient_order = np.argsort(explanation.reshape(-1, n_steps**2), axis=1)
        for i in range(n_steps+1):
            input_latents = torch.cat([start.cuda()] * 2)
            pred = self.unet(input_latents, t, encoder_hidden_states=encoder_hidden).sample
            noise_pred_uncond, noise_pred_text = pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            pred_latents = self.scheduler.step(noise_pred_cfg, t, start.cuda(), **self.extra_step_kwargs).prev_sample
            pred_img = self.decode_latents(pred_latents)
            pred = torch.from_numpy(pred_img).permute(0,3,1,2)
            score = pfw.fid(pred, target_img)
            # score = cos(pred, target_img)
            scores[i] = score.mean()
            # Render image if verbose, if it's the last step or if save is required.
            # if verbose == 0:
            #     if i == 0:
            #         plt.figure(figsize=(10, 5))
            #         plt.subplot(331)
            #         plt.title("Target image")
            #         plt.imshow(target_img.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off")
            #     elif i == 1:
            #         plt.subplot(332)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off")
            #     elif i == 3:
            #         plt.subplot(333)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off")
            #     elif i == 8:
            #         plt.subplot(334)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off")
            #     elif i == 30:
            #         plt.subplot(335)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off") 
            #     elif i == 40:
            #         plt.subplot(336)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off") 
            #     elif i == 50:
            #         plt.subplot(337)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off") 
            #     elif i == 70:
            #         plt.subplot(338)
            #         plt.title('D:{:.2f}%P={:.1f}'.format(100 * i / n_steps, scores[i]))
            #         plt.imshow(pred.squeeze(0).permute(1, 2, 0))
            #         plt.axis("off") 

            if i == n_steps:
                plt.figure(figsize=(10, 5))
                # plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                if verbose == 0:
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
                        plt.close()
                    else:
                        plt.show()
                return scores

            coords = salient_order[:, stride * i:stride * (i + 1)]
            start = start.cpu().numpy().reshape(1, 4, n_steps**2)

            start[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords]
            start = torch.from_numpy(start.reshape(1,4,n_steps,n_steps))
            # start.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords] = finish.cpu().numpy().reshape(1, 4, n_steps**2)[0, :, coords]
                # start.cpu().numpy().reshape(-1, n_steps**2)[:, coords] = finish.cpu().numpy().reshape(-1, n_steps**2)[:, coords]
        return scores


    @torch.no_grad()
    def lime_visualization(
                self, 
                target_latents, 
                input_latent,
                t, 
                hidden_states, 
                activation_map,
                layer_vis,
                get_auc_score,
                ):
        
        kernel_width=.25
        kernel=None
        verbose=False
        feature_selection='auto'
        random_state=None
        kernel_width = float(kernel_width)
        hide_color=0
        labels=(1,)
        num_samples=1000
        num_features=100000
        random_seed = None
        segmentation_fn = None
        
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        random_state = check_random_state(random_state)
        feature_selection = feature_selection

        base = LimeBase(kernel_fn, verbose, random_state=random_state)

        pdist = torch.nn.PairwiseDistance(p=2)

        h,w = target_latents.shape[2:]
        target_img = self.decode_latents(target_latents)
        target_img = torch.from_numpy(target_img).permute(0,3,1,2)


        transf = transforms.Compose([ transforms.Resize((h,w))])    
        perturb_image = np.transpose(np.array(transf(target_img.squeeze(0))), (1,2,0))

        if len(perturb_image.shape) == 2:
            perturb_image = gray2rgb(perturb_image)

        if random_seed is None:
            random_seed = random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        segments = segmentation_fn(perturb_image)

        fudged_image =torch.zeros_like(input_latent)

        top = labels
        n_features = np.unique(segments).shape[0]
        data = random_state.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        rows = data
        for row in tqdm(rows, desc = "Perturbing and decoding.."):
            temp = copy.deepcopy(input_latent)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(input_latent.shape[2:]).astype(bool)
            for z in zeros:
                mask[segments == z] = True

            temp[0,:,mask] = fudged_image[0,:,mask]

            input_latents = torch.cat([temp.cuda()] * 2)
            pred = self.unet(input_latents, t, encoder_hidden_states=hidden_states).sample
            noise_pred_uncond, noise_pred_text = pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            pred_latents = self.scheduler.step(noise_pred_cfg, t, temp.cuda(), **self.extra_step_kwargs).prev_sample
            # pred_img = self.decode_latents(pred_latents)
            # pred = torch.from_numpy(pred_img).permute(0,3,1,2)
            # score = pdist(target_img, pred)
            # labels.append(torch.Tensor([score.mean()]))
            labels.append(pred_latents.mean(1).reshape(1,-1).squeeze(0).cpu().numpy())
        labels = np.array(labels)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric='cosine'
        ).ravel()
        ret_exp = ImageExplanation(np.transpose(np.array(transf(target_img.squeeze(0))), (1,2,0)), segments)
        (ret_exp.intercept["score"],
        ret_exp.local_exp["score"],
        ret_exp.score["score"],
        ret_exp.local_pred["score"]) = base.explain_instance_with_data(
            data, labels, distances, 0, num_features,
            model_regressor=None,
            feature_selection="auto")
        temp, mask = self.get_image_and_mask(ret_exp, "score", positive_only = False, num_features=5, hide_rest=False)

        if get_auc_score:
            mode = "ins"
            auc_score = self.auc_run(mode, input_latent, target_latents, torch.from_numpy(temp.mean(2)).cuda(), t , hidden_states, random = False, verbose=0, save_to="outputs/auc/lime")
        else:
            auc_score = None

        return (temp, mask, ret_exp), auc_score


    def get_image_and_mask(self, ret_exp, label, positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in ret_exp.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = ret_exp.segments
        image = ret_exp.image
        exp = ret_exp.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(ret_exp.image.shape)
        else:
            temp = ret_exp.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                    if x[1].all() > 0 and x[1].all() > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                    if x[1].all() < 0 and abs(x[1].all()) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w.all()) < min_weight:
                    continue
                c = 0 if w.all() < 0 else 1
                mask[segments == f] = -1 if w.all() < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask