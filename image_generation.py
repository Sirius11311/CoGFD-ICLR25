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
from config import (FINETUNE_MODEL_DIR, GENERATED_IMG_DIR, PRETRAINED_MODEL_PATH, HARMFUL_CMB_DIR)

warnings.filterwarnings("ignore")



harmful_cmb_list = [
    "alcohol_and_driving",
    "muslim_and_alcohol",
    "muslim_and_pork",
    "pregnant_and_smoking",
    "underage_and_driving",
    "underage_and_weapon",
    "underage_and_alcohol",
    "underage_and_smoking",
    "underage_and_labor",
    "weapon_and_school"
]



def setup_directories(concept_combination, model_type):
    """Setup output directories for images and text files."""
    base_dir = f"{GENERATED_IMG_DIR.format(concept_combination=concept_combination)}/{model_type}"
    dirs = {
        'retain_img': f"{base_dir}/retain/img",
        'retain_txt': f"{base_dir}/retain/txt",
        'forget_img': f"{base_dir}/forget/img",
        'forget_txt': f"{base_dir}/forget/txt"
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def load_or_generate_prompts(concept_combination, sub_concepts, prompt_generate):
    """Load prompts from file or generate new ones if needed."""
    try:
        if concept_combination in harmful_cmb_list:
            with open(f"{HARMFUL_CMB_DIR}/{concept_combination}.json", 'r') as file:
                print(f"Loading prompts for {concept_combination}...")
                return json.load(file)
        else:
            with open(f"{GENERATED_IMG_DIR.format(concept_combination=concept_combination)}/{concept_combination}.json", 'r') as file:
                print(f"Loading prompts for {concept_combination}...")
                return json.load(file)
    except FileNotFoundError:
        if prompt_generate:
            print(f"there is no prompt file for {concept_combination}, generating prompts...")
            generate_image_prompts(concept_combination, sub_concepts, 
                                 f"{GENERATED_IMG_DIR.format(concept_combination=concept_combination)}/{concept_combination}.json")
            with open(f"{GENERATED_IMG_DIR.format(concept_combination=concept_combination)}/{concept_combination}.json", 'r') as file:
                return json.load(file)
        else:
            print(f"Prompt file not found for {concept_combination}. Please check the file path.")
            sys.exit()

def setup_pipeline(model_path, device="cuda:0"):
    """Initialize and setup the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    def dummy(images, **kwargs):
        return images, [False]
    pipe.safety_checker = dummy
    
    return pipe

def generate_and_save_images(pipe, data, concept, output_dirs, seed, args, is_forget=True):
    """Generate images for a given concept and save them."""
    data_list = data[concept]
    for cfg_text in args.cfg_text_list:
        for idx, prompt in enumerate(data_list):
            output_img_path = os.path.join( 
                output_dirs['forget_img' if is_forget else 'retain_img'],
                f"{concept}_{idx}_seed{seed}.jpg"
            )
            output_txt_path = os.path.join(
                output_dirs['forget_txt' if is_forget else 'retain_txt'],
                f"{concept}_{idx}_seed{seed}.txt"
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
            with open(output_txt_path, 'w') as f:
                f.write(prompt)

def main():
    parser = ArgumentParser()
    parser.add_argument("--concept_combination", required=True, type=str)
    parser.add_argument("--sub_concepts", required=False, default=[], type=str, nargs="+")
    parser.add_argument("--cfg-text-list", default=[9.0], nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=[188, 288, 588, 688, 888], nargs="+")
    parser.add_argument("--prompt_generate", action="store_true")
    parser.add_argument("--generate_ref", action="store_true")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--ddim_eta", type=float, default=0.0, 
                       help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--steps", default=100, type=int)

    args = parser.parse_args()

    args.ckpt_ft = FINETUNE_MODEL_DIR.format(concept_combination=args.concept_combination)
    args.ckpt_ref = PRETRAINED_MODEL_PATH
    
    if args.sub_concepts == []:
        args.sub_concepts = args.concept_combination.split("_and_")

    # Load or generate prompts
    data = load_or_generate_prompts(args.concept_combination, args.sub_concepts, args.prompt_generate)
    
    # Generate reference images if requested
    if args.generate_ref:
        print(f"Loading reference model from {args.ckpt_ref}...")
        pipe_ref = setup_pipeline(args.ckpt_ref)
        output_dirs = setup_directories(args.concept_combination, "ref")
        
        for seed in args.seed:
            seed_everything(seed)
            generate_and_save_images(pipe_ref, data, args.concept_combination, output_dirs, seed, args, is_forget=True)
            for sub_concept in args.sub_concepts:
                generate_and_save_images(pipe_ref, data, sub_concept, output_dirs, seed, args, is_forget=False)
    
    # Generate fine-tuned model images
    print(f"Loading fine-tuned model from {args.ckpt_ft}...")
    pipe = setup_pipeline(args.ckpt_ft)
    output_dirs = setup_directories(args.concept_combination, "ft")
    
    for seed in args.seed:
        seed_everything(seed)
        for sub_concept in args.sub_concepts:
            generate_and_save_images(pipe, data, sub_concept, output_dirs, seed, args, is_forget=False)
        generate_and_save_images(pipe, data, args.concept_combination, output_dirs, seed, args, is_forget=True)


if __name__ == "__main__":
    main()