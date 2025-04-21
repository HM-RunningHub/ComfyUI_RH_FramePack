# FramePack for ComfyUI
20250421 Update: Added support for first/last frame image-to-video generation from TTPlanetPig  
[TTPlanetPig](https://github.com/TTPlanetPig) https://github.com/lllyasviel/FramePack/pull/167 

## Online Access
You can access RunningHub online to use this plugin and models for free:
### English Version
- **Run & Download Workflow**:  
  [https://www.runninghub.ai/post/1912930457355517954](https://www.runninghub.ai/post/1912930457355517954)
### 中文版本
- **运行并下载工作流**:  
  [https://www.runninghub.cn/post/1912930457355517954](https://www.runninghub.cn/post/1912930457355517954)

## Features  
This is a simple implementation of https://github.com/lllyasviel/FramePack. If there are any advantages, they would be:  
- Better automatic adaptation for 24GB GPUs, enabling higher resolution processing whenever possible.  
- The entire workflow requires no parameter adjustments, making it extremely user-friendly.  




# Model Download Guide

## Choose a Download Method (Pick One)

1. **Download via Cloud Storage (for users in China)**
   - [T8模型包] (https://pan.quark.cn/s/9669ce6c7356)
2. **One-Click Download with Python Script**
   ```python
   from huggingface_hub import snapshot_download

   # Download HunyuanVideo model
   snapshot_download(
       repo_id="hunyuanvideo-community/HunyuanVideo",
       local_dir="HunyuanVideo",
       ignore_patterns=["transformer/*", "*.git*", "*.log*", "*.md"],
       local_dir_use_symlinks=False
   )

   # Download flux_redux_bfl model
   snapshot_download(
       repo_id="lllyasviel/flux_redux_bfl",
       local_dir="flux_redux_bfl",
       ignore_patterns=["*.git*", "*.log*", "*.md"],
       local_dir_use_symlinks=False
   )

   # Download FramePackI2V_HY model
   snapshot_download(
       repo_id="lllyasviel/FramePackI2V_HY",
       local_dir="FramePackI2V_HY",
       ignore_patterns=["*.git*", "*.log*", "*.md"],
       local_dir_use_symlinks=False
   )

3. **Manual Download**
   - HunyuanVideo: [HuggingFace Link](https://huggingface.co/hunyuanvideo-community/HunyuanVideo/tree/main)
   - Flux Redux BFL: [HuggingFace Link](https://huggingface.co/lllyasviel/flux_redux_bfl/tree/main)
   - FramePackI2V: [HuggingFace Link](https://huggingface.co/lllyasviel/FramePackI2V_HY/tree/main)

4. **File Structure After Download**
```
comfyui/models/
  flux_redux_bfl
  ├── feature_extractor
  │   └── preprocessor_config.json
  ├── image_embedder
  │   ├── config.json
  │   └── diffusion_pytorch_model.safetensors
  ├── image_encoder
  │   ├── config.json
  │   └── model.safetensors
  ├── model_index.json
  └── README.md
  FramePackI2V_HY
  ├── config.json
  ├── diffusion_pytorch_model-00001-of-00003.safetensors
  ├── diffusion_pytorch_model-00002-of-00003.safetensors
  ├── diffusion_pytorch_model-00003-of-00003.safetensors
  ├── diffusion_pytorch_model.safetensors.index.json
  └── README.md
  HunyuanVideo
  ├── config.json
  ├── model_index.json
  ├── README.md
  ├── scheduler
  │   └── scheduler_config.json
  ├── text_encoder
  │   ├── config.json
  │   ├── model-00001-of-00004.safetensors
  │   ├── model-00002-of-00004.safetensors
  │   ├── model-00003-of-00004.safetensors
  │   ├── model-00004-of-00004.safetensors
  │   └── model.safetensors.index.json
  ├── text_encoder_2
  │   ├── config.json
  │   └── model.safetensors
  ├── tokenizer
  │   ├── special_tokens_map.json
  │   ├── tokenizer_config.json
  │   └── tokenizer.json
  ├── tokenizer_2
  │   ├── merges.txt
  │   ├── special_tokens_map.json
  │   ├── tokenizer_config.json
  │   └── vocab.json
  └── vae
      ├── config.json
      └── diffusion_pytorch_model.safetensors
```
## Example:


https://github.com/user-attachments/assets/4378bb8c-a8f4-4f16-a835-cde976c6144e


![image](https://github.com/user-attachments/assets/ea936caf-c0ca-48f4-af20-64090771d382)

