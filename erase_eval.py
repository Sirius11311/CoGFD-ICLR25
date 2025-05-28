import subprocess
import argparse
from pprint import pprint
import json
import os
from config import (GENERATED_IMG_DIR, EVAL_RESULTS_DIR)

def get_clip_score(img_folder, txt_folder, model='ViT-B/32'):
    cmd = [
        "python", "-m", "clip_score",
        "--clip-model", model,
        img_folder,
        txt_folder
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout + "\n" + result.stderr
    # print("clip_score output:\n", output)  # 调试用
    for line in output.splitlines():
        if "CLIP Score:" in line:
            score_str = line.split("CLIP Score:")[-1].strip()
            try:
                return float(score_str)
            except ValueError:
                pass
    raise RuntimeError("CLIP Score not found in output:\n" + output)

def get_clip_score_list(concept_combination, file_path, output_dir):
    model_type_list = ['ref', 'ft']
    concept_type_list = ['forget', 'retain']
    clip_score_dic = {}

    # First collect all scores
    for model_type in model_type_list:
        clip_score_dic[model_type] = {}
        for concept_type in concept_type_list:
            img_folder = file_path + '/' + model_type + '/' + concept_type + '/img'
            txt_folder = file_path + '/' + model_type + '/' + concept_type + '/txt'
            score = get_clip_score(img_folder, txt_folder)
            print(f"CLIP Score of {concept_combination} {concept_type} in {model_type} model:", score)
            clip_score_dic[model_type][concept_type] = score

    # Calculate erase_retain_score
    forget_change = clip_score_dic['ref']['forget'] - clip_score_dic['ft']['forget']
    retain_change = clip_score_dic['ref']['retain'] - clip_score_dic['ft']['retain']
    erase_retain_score = abs(forget_change / retain_change) if retain_change != 0 else float('inf')
    clip_score_dic['erase_retain_score'] = erase_retain_score
    print(f"Erase-Retain Score: {erase_retain_score}")

    # Save results to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{concept_combination}.json')
    
    with open(output_file, 'w') as f:
        json.dump(clip_score_dic, f, indent=4)
    print(f"Results saved to {output_file}")

    pprint(clip_score_dic)
    return clip_score_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_combination', type=str, required=True)
    args = parser.parse_args()
    file_path = GENERATED_IMG_DIR.format(concept_combination=args.concept_combination)
    output_dir = EVAL_RESULTS_DIR.format(concept_combination=args.concept_combination)


    get_clip_score_list(args.concept_combination, file_path, output_dir)
    
