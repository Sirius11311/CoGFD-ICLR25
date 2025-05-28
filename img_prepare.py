import os.path
import os
import torch
from diffusers import StableDiffusionPipeline
from argparse import ArgumentParser
import sys
from pytorch_lightning import seed_everything
import json
from prompt_generator import generate_image_prompts
import warnings
from config import (PRETRAINED_MODEL_PATH, PREPARED_DATA_DIR)

warnings.filterwarnings("ignore")

def setup_pipeline(model_path, device="cuda:0"):
    """Initialize and setup the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    def dummy(images, **kwargs):
        return images, [False]
    pipe.safety_checker = dummy
    
    return pipe

def generate_and_save_images(pipe, data, concept, output_dirs, seed, args):
    """Generate images for a given concept and save them."""
    for cfg_text in args.cfg_text_list:
        for idx, prompt in enumerate(data):
            output_img_path = os.path.join( 
                output_dirs, f"{concept}_{idx}_seed{seed}.jpg"
            )
            if os.path.exists(output_img_path):
                print(f"Detected! Skipping: {output_img_path}!")
                continue
                
            print(f"Generating: {prompt}")
            image = pipe(
                prompt=prompt,
                width=args.W,
                height=args.H,
                num_inference_steps=args.steps,
                guidance_scale=cfg_text
            ).images[0]
            
            image.save(output_img_path)

def main():
    parser = ArgumentParser()
    parser.add_argument("--concept_combination", required=True, type=str)
    parser.add_argument("--cfg-text-list", default=[9.0], nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=[188, 288, 588, 688, 888], nargs="+")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--ddim_eta", type=float, default=0.0, 
                       help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--steps", default=100, type=int)

    args = parser.parse_args()

    args.ckpt_ref = PRETRAINED_MODEL_PATH
    
    data = [f'an image of  {args.concept_combination}']
    
    print(f"Loading reference model from {args.ckpt_ref}...")
    pipe_ref = setup_pipeline(args.ckpt_ref)
    output_dir = f"{PREPARED_DATA_DIR.format(concept_combination=args.concept_combination)}"
    os.makedirs(output_dir, exist_ok=True)
    
    for seed in args.seed:
        seed_everything(seed)
        generate_and_save_images(pipe_ref, data, args.concept_combination, output_dir, seed, args)

    
if __name__ == "__main__":
    main()