from diffusers import StableDiffusionPipeline
import torch
from attn_processor import AttnProcessor
import argparse
from utils import overlay_attention_map, save_image
import numpy as np
import cv2
import os
import math
import shutil

import re

def safe_filename(s):
    # 替换Windows非法字符为空下划线
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', s)


def main(prompt, output_path, attn_dir="attn_frames", skip_empty_token=True):
    # 加载模型
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # 注入自定义AttnProcessor，只hook交叉注意力层
    attn_processor = AttnProcessor()
    # target_layer = "mid_block.attentions.0.transformer_blocks.0.attn2"
    target_layer = "up_blocks.3.attentions.2.transformer_blocks.0.attn2"
    for name, module in pipe.unet.named_modules():
        if name == target_layer and hasattr(module, "set_processor"):
            module.set_processor(attn_processor)

    # 生成图片并收集注意力图
    result = pipe(prompt)
    image = result.images[0]
    attention_maps = attn_processor.attention_maps
    print(f"attn_maps.shape:{len(attention_maps)}")
    print(f"attn_maps[0].shape:{attention_maps[0].shape}")

    # 获取tokenizer和token文本
    tokenizer = pipe.tokenizer
    tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    token_ids = tokens.input_ids[0].cpu().numpy()
    token_strs = [tokenizer.decode([i]).strip().replace('/', '_') for i in token_ids]

    # 创建保存注意力图的文件夹
    if os.path.exists(attn_dir):
            shutil.rmtree(attn_dir)
    os.makedirs(attn_dir, exist_ok=True)
    image_np = np.array(image)

    for idx, attn_map in enumerate(attention_maps):
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.float().cpu().numpy()
        # 对head做平均，得到 [40, 77]
        if attn_map.ndim == 3:
            attn_map = attn_map.mean(axis=0)  # [40, 77]
        tokens_num = attn_map.shape[0]
        if tokens_num == 40:
            h, w = 5, 8
        elif tokens_num == 4096:
            h, w = 64, 64
        else:
            h = int(math.sqrt(tokens_num))
            w = tokens_num // h
        for token_idx, token_text in enumerate(token_strs):
            # 跳过空token
            if skip_empty_token and (not token_text or token_text == "!"):
                continue
            spatial_map = attn_map[:, token_idx]  # [40]
            spatial_map = spatial_map.reshape(h, w)
            spatial_map = spatial_map - spatial_map.min()
            if spatial_map.max() > 0:
                spatial_map = spatial_map / spatial_map.max()
            spatial_map = (spatial_map * 255).astype(np.uint8)
            spatial_map_resized = cv2.resize(spatial_map, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            # 为每个token单独建文件夹，token_text做合法化
            safe_token_text = safe_filename(token_text)
            token_dir = os.path.join(attn_dir, f"token_{token_idx:02d}_{safe_token_text}")
            os.makedirs(token_dir, exist_ok=True)
            save_path = os.path.join(token_dir, f"attn_step_{idx:03d}.png")
            overlay = overlay_attention_map(image_np, spatial_map_resized)
            save_image(overlay, save_path)

    # 也可以保存最终图片
    save_image(image_np, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images and visualize attention maps.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate images.")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the output image.")
    parser.add_argument("--attn_dir", type=str, default="attn_frames", help="Directory to save attention maps.")
    parser.add_argument("--no_skip_empty_token",default=False, action="store_true", help="Do not skip empty tokens when saving attention maps.")
    args = parser.parse_args()

    main(
        args.prompt,
        args.output_path,
        args.attn_dir,
        skip_empty_token=not args.no_skip_empty_token
    )