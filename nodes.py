import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import hashlib
import random
import string
import torchvision
from torchvision.transforms.functional import to_pil_image
import comfy.utils

from PIL import Image
import folder_paths

class Kiki_FramePack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True}),
                # "n_prompt": ("STRING", {"multiline": True}),
                "total_second_length": ("INT", {"default": 5, "min": 1, "max": 120, "step": 1}),
                "seed": ("INT", {"default": 3407}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "use_teacache": ("BOOLEAN", {"default": True}),
                "upscale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.0, "step": 0.1, "description": "Resolution scaling factor. 1.0 = original size, >1.0 = upscale, <1.0 = downscale"}),
            },
            "optional": {
                "end_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    CATEGORY = "Runninghub/FramePack"
    FUNCTION = "run"

    TITLE = 'RunningHub FramePack'
    OUTPUT_NODE = True

    def __init__(self):
        self.high_vram = False
        self.frames = None
        self.fps = None

        hunyuan_root = os.path.join(folder_paths.models_dir, 'HunyuanVideo')
        flux_redux_bfl_root = os.path.join(folder_paths.models_dir, 'flux_redux_bfl')
        framePackI2V_root = os.path.join(folder_paths.models_dir, 'FramePackI2V_HY')

        self.text_encoder = LlamaModel.from_pretrained(hunyuan_root, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(hunyuan_root, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hunyuan_root, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(hunyuan_root, subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(hunyuan_root, subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained(flux_redux_bfl_root, subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained(flux_redux_bfl_root, subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(framePackI2V_root, torch_dtype=torch.bfloat16).cpu()

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        self.transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')

        self.transformer.to(dtype=torch.bfloat16)
        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if not self.high_vram:
            # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)

    def strict_align(self, h, w, scale):
        raw_h = h * scale
        raw_w = w * scale

        aligned_h = int(round(raw_h / 64)) * 64
        aligned_w = int(round(raw_w / 64)) * 64

        assert (aligned_h % 64 == 0) and (aligned_w % 64 == 0), "尺寸必须是64的倍数"
        assert (aligned_h//8) % 8 == 0 and (aligned_w//8) % 8 == 0, "潜在空间需要8的倍数"
        return aligned_h, aligned_w

    def preprocess_image(self, image):
        if image is None:
            return None
        image_np = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8)).convert("RGB")
        input_image = np.array(image)
        return input_image

    def run(self, **kwargs):
        try:
            image = kwargs['ref_image']
            end_image = kwargs.get('end_image', None)  # Use get with None as default
            image_np = self.preprocess_image(image)
            end_image_np = self.preprocess_image(end_image) if end_image is not None else None
            prompt = kwargs['prompt']
            seed = kwargs['seed']
            total_second_length = kwargs['total_second_length']
            steps = kwargs['steps']
            use_teacache = kwargs['use_teacache']
            upscale = kwargs['upscale']
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            video_path = os.path.join(folder_paths.get_output_directory(), f'{random_str}.mp4')

            self.pbar = comfy.utils.ProgressBar(steps * total_second_length)

            self.exec(input_image=image_np, end_image=end_image_np, prompt=prompt, seed=seed, total_second_length=total_second_length, video_path=video_path, steps=steps, use_teacache=use_teacache, scale=upscale)
            
            if os.path.exists(video_path):
                self.fps = self.get_fps_with_torchvision(video_path)
                self.frames = self.extract_frames_as_pil(video_path)
                print(f'{video_path}:{self.fps} {len(self.frames)}')
            else:
                self.frames = []
                self.fps = 0.0
        except Exception as e:
            print(f"Error in run: {str(e)}")
            traceback.print_exc()
            self.frames = []
            self.fps = 0.0

        return (self.frames, self.fps)
        
    @torch.no_grad()
    def exec(self, input_image, video_path,
            end_image=None,
            prompt="The girl dances gracefully, with clear movements, full of charm.", 
            n_prompt="", 
            seed=31337, 
            total_second_length=5, 
            latent_window_size=9, 
            steps=25, 
            cfg=1, 
            gs=32, 
            rs=0, 
            gpu_memory_preservation=6, 
            use_teacache=True,
            scale=1.0):
        
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        try:
            # Clean GPU
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer
                )

            # Text encoding
            print('Text encoding')

            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, gpu)
                load_model_as_complete(self.text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Processing input image (start frame)
            print('Processing start frame ...')

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            print(f"Resized height: {height}, Resized width: {width}")
            
            height, width = self.strict_align(height, width, scale)
            print(f"After Resized height: {height}, Resized width: {width}")
            
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # Processing end image if provided
            has_end_image = end_image is not None
            end_image_np = None
            end_image_pt = None
            
            if has_end_image:
                print('Processing end frame ...')
                H_end, W_end, C_end = end_image.shape
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            print('VAE encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, self.vae)
            end_latent = None
            if has_end_image:
                end_latent = vae_encode(end_image_pt, self.vae)

            # CLIP Vision
            print('CLIP Vision encoding ...')

            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)

            # Start image encoding
            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            # End image encoding if available
            if has_end_image:
                end_image_encoder_output = hf_clip_vision_encode(end_image_np, self.feature_extractor, self.image_encoder)
                end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
                # Use a simple average of embeddings - exactly like in the original code
                image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

            # Dtype
            llama_vec = llama_vec.to(self.transformer.dtype)
            llama_vec_n = llama_vec_n.to(self.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)

            print('Start Sample')

            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = list(reversed(range(total_latent_sections)))

            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for i, latent_padding in enumerate(latent_paddings):
                is_last_section = latent_padding == 0
                is_first_section = latent_padding == latent_paddings[0]  # Use the original method
                latent_padding_size = latent_padding * latent_window_size

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # Always use start_latent for the first position (exactly like in the original code)
                clean_latents_pre = start_latent.to(history_latents)
                
                # For the second position, use history
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                
                # Create clean_latents first
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
                # Then if we have end_image and this is the first section, override clean_latents_post with end_latent
                if has_end_image and is_first_section:
                    clean_latents_post = end_latent.to(history_latents)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    self.update(1)
                    return

                generated_latents = sample_hunyuan(
                    transformer=self.transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                # For the last section, add start_latent back to the beginning - just like in the original
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                # Accumulate generated frames
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=gpu)

                # Only decode up to the total number of frames we've generated
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                # Decode latents to pixels
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, self.vae).cpu()
                else:
                    # For appending new frames to existing ones
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                if not self.high_vram:
                    unload_complete_models()

                # If this is the last section, save the video
                if is_last_section:
                    save_bcthw_as_mp4(history_pixels, video_path, fps=30)
                    break

        except Exception as e:
            print(f"Error in exec: {str(e)}")
            traceback.print_exc()
        finally:
            unload_complete_models()
        
    def update(self, in_progress):
        self.pbar.update(in_progress)

    def extract_frames_as_pil(self, video_path):
        video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')  # (T, H, W, C)
        frames = [to_pil_image(frame.permute(2, 0, 1)) for frame in video]
        frames = [torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in frames]
        return frames               

    def get_fps_with_torchvision(self, video_path):
        _, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        return info['video_fps']

# --- Start of Kiki_FramePack_F1 Class ---
class Kiki_FramePack_F1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True}),
                "total_second_length": ("INT", {"default": 5, "min": 1, "max": 120, "step": 1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60, "step": 1}),
                "seed": ("INT", {"default": 3407}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "gs": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 32.0, "step": 0.1, "round": 0.01, "label": "Distilled CFG Scale"}),
                "use_teacache": ("BOOLEAN", {"default": True}),
                "upscale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "description": "Resolution scaling factor."}),
            },
            "optional": {
                 "n_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    CATEGORY = "Runninghub/FramePack"
    FUNCTION = "run_f1"

    TITLE = 'RunningHub FramePack F1'
    OUTPUT_NODE = True

    def __init__(self):
        self.high_vram = False
        self.frames = None
        self.fps = None

        hunyuan_root = os.path.join(folder_paths.models_dir, 'HunyuanVideo')
        flux_redux_bfl_root = os.path.join(folder_paths.models_dir, 'flux_redux_bfl')
        framePackF1_root = os.path.join(folder_paths.models_dir, 'FramePackF1_HY')
        
        if not os.path.isdir(framePackF1_root):
             print(f"Warning: FramePack F1 model directory not found at {framePackF1_root}")

        self.text_encoder = LlamaModel.from_pretrained(hunyuan_root, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(hunyuan_root, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(hunyuan_root, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(hunyuan_root, subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(hunyuan_root, subfolder='vae', torch_dtype=torch.float16).cpu()

        self.feature_extractor = SiglipImageProcessor.from_pretrained(flux_redux_bfl_root, subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained(flux_redux_bfl_root, subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        try:
            self.transformer_f1 = HunyuanVideoTransformer3DModelPacked.from_pretrained(framePackF1_root, torch_dtype=torch.bfloat16).cpu()
        except Exception as e:
             print(f"Error loading FramePack F1 transformer model from {framePackF1_root}: {e}")
             print("Please ensure the F1 model weights (e.g., transformer.safetensors) are correctly placed in the directory.")
             self.transformer_f1 = None

        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        if self.transformer_f1:
             self.transformer_f1.eval()

        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        if self.transformer_f1:
             self.transformer_f1.high_quality_fp32_output_for_inference = True
             print('F1 transformer.high_quality_fp32_output_for_inference = True')

             self.transformer_f1.to(dtype=torch.bfloat16)

             self.transformer_f1.requires_grad_(False)

             if not self.high_vram:
                 DynamicSwapInstaller.install_model(self.transformer_f1, device=gpu)

        self.vae.to(dtype=torch.float16)
        self.image_encoder.to(dtype=torch.float16)
        self.text_encoder.to(dtype=torch.float16)
        self.text_encoder_2.to(dtype=torch.float16)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        if not self.high_vram:
             DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)

    def strict_align(self, h, w, scale):
        raw_h = h * scale
        raw_w = w * scale
        aligned_h = int(round(raw_h / 64)) * 64
        aligned_w = int(round(raw_w / 64)) * 64
        assert (aligned_h % 64 == 0) and (aligned_w % 64 == 0), "尺寸必须是64的倍数"
        assert (aligned_h//8) % 8 == 0 and (aligned_w//8) % 8 == 0, "潜在空间需要8的倍数"
        return aligned_h, aligned_w

    def preprocess_image(self, image):
        if image is None: return None
        if image.dim() == 4 and image.shape[0] == 1:
             img_tensor = image[0]
        else:
             img_tensor = image 
             print(f"Warning: Unexpected input image tensor shape: {image.shape}. Assuming HWC.")
             
        image_np = 255. * img_tensor.cpu().numpy()
        image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8)).convert("RGB")
        input_image = np.array(image)
        return input_image
        
    def run_f1(self, **kwargs):
        if not self.transformer_f1:
            print("Error: Kiki_FramePack_F1 cannot run because the transformer model failed to load.")
            return (torch.empty((0, 1, 1, 3), dtype=torch.float32), 0.0)

        try:
            image = kwargs['ref_image']
            image_np = self.preprocess_image(image)
            prompt = kwargs['prompt']
            n_prompt = kwargs.get('n_prompt', "")
            seed = kwargs['seed']
            total_second_length = kwargs['total_second_length']
            fps = kwargs['fps']
            steps = kwargs['steps']
            gs = kwargs['gs']
            use_teacache = kwargs['use_teacache']
            upscale = kwargs['upscale']
            cfg = 1.0
            rs = 0.0
            latent_window_size = 9

            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            video_path = os.path.join(folder_paths.get_output_directory(), f'{random_str}_f1.mp4')

            # --- Initialize Progress Bar (Aligned with demo's section calc) ---
            # Use demo's calculation for total_latent_sections, assuming 30fps basis for consistency
            total_latent_sections = int(max(round((total_second_length * 30) / (latent_window_size * 4)), 1))
            total_progress_steps = total_latent_sections * steps
            self.pbar = comfy.utils.ProgressBar(total_progress_steps)

            # Call exec_f1, passing latent_window_size as well
            self.exec_f1(input_image=image_np, prompt=prompt, n_prompt=n_prompt, seed=seed,
                         total_second_length=total_second_length, video_path=video_path, fps=fps,
                         steps=steps, gs=gs, cfg=cfg, rs=rs, latent_window_size=latent_window_size, # Pass latent_window_size
                         use_teacache=use_teacache, scale=upscale,
                         gpu_memory_preservation=6)

            if os.path.exists(video_path):
                self.fps = float(fps)
                self.frames = self.extract_frames_to_tensor(video_path)
                print(f'F1 Video saved: {video_path} | FPS: {self.fps} | Frames: {self.frames.shape[0] if self.frames is not None else 0}')
            else:
                self.frames = torch.empty((0, 1, 1, 3), dtype=torch.float32)
                self.fps = 0.0
                print(f'F1 Video generation failed or file not found: {video_path}')

        except Exception as e:
            print(f"Error in run_f1: {str(e)}")
            traceback.print_exc()
            self.frames = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            self.fps = 0.0

        return (self.frames, self.fps)

    @torch.no_grad()
    def exec_f1(self, input_image, video_path,
                prompt, n_prompt, seed, total_second_length, fps,
                steps, gs, cfg, rs, latent_window_size, # Receive latent_window_size
                use_teacache, scale,
                gpu_memory_preservation=6):

        print("--- Starting Kiki_FramePack_F1 exec_f1 (Aligned with Demo Logic) ---")
        print(f"Params: seed={seed}, length={total_second_length}s@{fps}fps, steps={steps}, gs={gs}, cfg={cfg}, rs={rs}, lws={latent_window_size}")

        vae_time_stride = 4

        # --- Use Demo's total_latent_sections calculation --- 
        total_latent_sections = int(max(round((total_second_length * 30) / (latent_window_size * 4)), 1))
        print(f"Total generation sections (Demo calc): {total_latent_sections}")

        # --- Calculate target frames needed (still useful for trimming) ---
        target_pixel_frames = int(round(total_second_length * fps))

        try:
            # --- 1. Initialization & Setup --- 
            torch.manual_seed(seed)
            rnd = torch.Generator("cpu").manual_seed(seed)

            # ... (Unload models if needed) ...

            # --- 2. Encoding Inputs --- 
            print('Encoding text prompts...')
            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, gpu)
                load_model_as_complete(self.text_encoder_2, target_device=gpu)
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            print('Processing reference image...')
            H, W, C = input_image.shape
            if scale == 1.0:
                 height, width = find_nearest_bucket(H, W, resolution=640)
                 height, width = self.strict_align(height, width, 1.0)
            else:
                 height, width = self.strict_align(H, W, scale)
            print(f"Target dimensions: {width}x{height}")
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            print('VAE encoding reference image...')
            if not self.high_vram: load_model_as_complete(self.vae, target_device=gpu)
            start_latent = vae_encode(input_image_pt.to(self.vae.device, dtype=self.vae.dtype), self.vae)
            print(f"Start latent shape: {start_latent.shape}")

            print('CLIP Vision encoding reference image...')
            if not self.high_vram: load_model_as_complete(self.image_encoder, target_device=gpu)
            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder.to(gpu))
            image_embeddings = image_encoder_output.last_hidden_state

            transformer_dtype = self.transformer_f1.dtype
            start_latent = start_latent.to(transformer_dtype).cpu()

            # --- 3. Diffusion Loop (Aligned with Demo) --- 
            print(f'Starting diffusion loop for {total_latent_sections} sections...')

            latent_channels = start_latent.shape[1]
            latent_height = start_latent.shape[-2]
            latent_width = start_latent.shape[-1]
            history_context_size = 16 + 2 + 1

            # --- Initialize history_latents like demo --- 
            # Start with zeros matching context size
            history_latents = torch.zeros(size=(1, latent_channels, history_context_size, latent_height, latent_width), dtype=torch.float32).cpu() # Use float32 like demo?
            # Immediately add start_latent
            history_latents = torch.cat([history_latents, start_latent.to(history_latents.dtype)], dim=2)
            total_generated_latent_frames = 1 # Account for start_latent
            history_pixels = None

            # ... (Progress bar callback setup) ...
            current_section_step = 0
            total_progress_steps = total_latent_sections * steps
            def callback_f1(d):
                 # ... (Update pbar logic remains the same) ...
                 nonlocal current_section_step
                 step_in_section = d['i']
                 current_total_step = current_section_step * steps + step_in_section + 1
                 if hasattr(self, 'pbar') and self.pbar:
                     self.pbar.update_absolute(current_total_step, total_progress_steps)

            # Calculate frames generated per step based on demo
            frames_per_latent_window = latent_window_size * 4 - 3

            for section_index in range(total_latent_sections):
                section_start_time = time.time()
                print(f'Generating section {section_index + 1} / {total_latent_sections}')
                current_section_step = section_index

                # ... (Load transformer if needed) ...

                # --- Prepare context and indices (same as before, uses history_latents) --- 
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                # Get history context from the *end* of the current history_latents
                # No padding needed here because history starts with context + start_latent
                history_context = history_latents[:, :, -history_context_size:, :, :]
                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_context.split([16, 2, 1], dim=2)
                clean_latents = torch.cat([start_latent.cpu(), clean_latents_1x.cpu()], dim=2)

                # --- Prepare sample_kwargs (same as before) --- 
                sample_kwargs = dict(
                    transformer=self.transformer_f1,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=frames_per_latent_window, # Use demo's frame count
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    # --- Add missing positive prompt embeddings & ENSURE DTYPE --- 
                    prompt_embeds=llama_vec.to(gpu, dtype=transformer_dtype),
                    prompt_embeds_mask=llama_attention_mask.to(gpu), # Mask dtype usually okay
                    # --- Existing embeddings/poolers & ENSURE DTYPE --- 
                    prompt_poolers=clip_l_pooler.to(gpu, dtype=transformer_dtype),
                    negative_prompt_embeds=llama_vec_n.to(gpu, dtype=transformer_dtype),
                    negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu), # Mask dtype usually okay
                    negative_prompt_poolers=clip_l_pooler_n.to(gpu, dtype=transformer_dtype),
                    device=gpu, # Device is already GPU
                    dtype=transformer_dtype, # Explicitly passing transformer's dtype
                    image_embeddings=image_embeddings.to(gpu, dtype=transformer_dtype),
                    latent_indices=latent_indices.to(gpu), # Indices dtype usually okay
                    clean_latents=clean_latents.to(gpu, dtype=transformer_dtype), # Ensure correct dtype
                    clean_latent_indices=clean_latent_indices.to(gpu), # Indices dtype usually okay
                    clean_latents_2x=clean_latents_2x.to(gpu, dtype=transformer_dtype), # Ensure correct dtype
                    clean_latent_2x_indices=clean_latent_2x_indices.to(gpu), # Indices dtype usually okay
                    clean_latents_4x=clean_latents_4x.to(gpu, dtype=transformer_dtype), # Ensure correct dtype
                    clean_latent_4x_indices=clean_latent_4x_indices.to(gpu), # Indices dtype usually okay
                    callback=callback_f1,
                )

                # ... (Initialize teacache) ...
                if hasattr(self.transformer_f1, 'initialize_teacache'):
                    self.transformer_f1.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

                # --- Call sample_hunyuan --- 
                generated_latents = sample_hunyuan(**sample_kwargs)

                generated_latents = generated_latents.to(cpu, dtype=torch.float32)
                print(f"  Sampled latent section shape: {generated_latents.shape}")

                # --- Update history_latents (Aligned with Demo: Always append) --- 
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents.dtype)], dim=2)

                # --- Decode and append pixels (Aligned with Demo) --- 
                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer_f1, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=gpu)
                else:
                     if self.vae.device != gpu: self.vae.to(gpu)

                # Calculate the slice of history to decode based on total generated frames
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :] # Use actual generated frames

                if history_pixels is None:
                    # First time: decode the current relevant history
                    history_pixels = vae_decode(real_history_latents.to(gpu, dtype=self.vae.dtype), self.vae).cpu()
                    print(f"  Decoded initial pixels. Shape: {history_pixels.shape}")
                else:
                    # Subsequent times: decode only the part needed for smooth append
                    section_latent_frames = latent_window_size * 2
                    overlapped_frames = latent_window_size * 4 - 3 # Use demo's overlap calculation

                    # Decode the relevant tail end of the history latents
                    current_latents_to_decode = real_history_latents[:, :, -section_latent_frames:, :, :]
                    current_pixels = vae_decode(current_latents_to_decode.to(gpu, dtype=self.vae.dtype), self.vae).cpu()

                    # Append smoothly using demo's overlap value
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
                    print(f"  Appended pixels. New history shape: {history_pixels.shape}")

                # ... (Unload VAE if needed) ...
                if not self.high_vram:
                    unload_complete_models(self.vae)

                section_end_time = time.time()
                print(f"  Section {section_index + 1} took {section_end_time - section_start_time:.2f} seconds.")

            # --- 4. Final Saving (Aligned with Demo, keeping variable fps) --- 
            print('Saving final video...')
            if history_pixels is None or history_pixels.shape[2] == 0:
                 raise ValueError("No pixel frames were generated or decoded.")

            if history_pixels.shape[2] > target_pixel_frames:
                 print(f"Trimming final video from {history_pixels.shape[2]} to {target_pixel_frames} frames.")
                 history_pixels = history_pixels[:,:,:target_pixel_frames,:,:]

            save_bcthw_as_mp4(
                history_pixels,
                video_path,
                fps=fps, # Keep user FPS for now
                # crf=18 # Omit crf until utils.py is confirmed synced
            )
            print(f"Final video saved to: {video_path}")

        except Exception as e:
            print(f"Error during Kiki_FramePack_F1 execution: {str(e)}")
            traceback.print_exc()
            if os.path.exists(video_path):
                try: os.remove(video_path)
                except OSError: pass
            if hasattr(self, 'pbar') and self.pbar: self.pbar.update_absolute(total_progress_steps, total_progress_steps)
            raise

        finally:
            print('Cleaning up models...')
            unload_complete_models(
                self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer_f1
            )
            torch.cuda.empty_cache()
            print("--- Finished Kiki_FramePack_F1 exec_f1 (Aligned with Demo Logic) ---")

    def extract_frames_to_tensor(self, video_path):
        try:
            video_tensor, _, metadata = torchvision.io.read_video(video_path, pts_unit='sec', output_format='TCHW')
            
            video_tensor = video_tensor.permute(0, 2, 3, 1)
            
            video_tensor = video_tensor.float() / 255.0
            
            print(f"Extracted video tensor shape: {video_tensor.shape}")
            return video_tensor

        except Exception as e:
            print(f"Error extracting frames using torchvision.io.read_video: {e}")
            traceback.print_exc()
            return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    def get_fps_with_torchvision(self, video_path):
        try:
            _, _, metadata = torchvision.io.read_video(video_path, pts_unit='sec')
            fps = metadata.get('video_fps', 30.0)
            return float(fps)
        except Exception as e:
            print(f"Error reading FPS using torchvision.io.read_video: {e}")
            traceback.print_exc()
            return 30.0

# NODE CLASS MAPPINGS
NODE_CLASS_MAPPINGS = {
    "RunningHub_FramePack": Kiki_FramePack,
    "RunningHub_FramePack_F1": Kiki_FramePack_F1
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_FramePack": Kiki_FramePack.TITLE,
    "RunningHub_FramePack_F1": Kiki_FramePack_F1.TITLE
}
