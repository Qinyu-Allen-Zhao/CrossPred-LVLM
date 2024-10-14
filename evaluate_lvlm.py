import os
import json
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset
from utils.func import split_list, get_chunk

from utils.metric import Evaluator


def get_model_output(args, chunk_idx, data, extra_keys):
    model = build_model(args)

    print("Getting model output ...")
    for i, idx in tqdm(enumerate(chunk_idx), total=len(chunk_idx)):
        if os.path.exists(os.path.join(args.store_path, f"{idx}.pth")):
            print(f"Skip {idx}")
            continue

        ins = data[i]
        images = ins['images']
        prompt = ins['question']

        response, _, logits, _, hidden_states = model.forward_with_probs(images, prompt)
        
        if args.only_ans:
            out = {
                "question": ins["question"],
                "category": ins['category'],
                "response": response,
                "label": ins['label']
            }
        else:
            out = {
                "question": ins["question"],
                "category": ins['category'],
                "hidden_states": hidden_states,
                "response": response,
                "logits": logits[[0]],
                "label": ins['label']
            }
        for k in extra_keys:
            out[k] = ins[k]

        torch.save(out, os.path.join(args.store_path, f"{idx}.pth"))


def eval_model_output(args, chunk_idx, data):
    print("Evaluating model output ...")
    evaluator = Evaluator()
    
    for i, idx in tqdm(enumerate(chunk_idx), total=len(chunk_idx)):
        ins = data[i]
        file_name = os.path.join(args.store_path, f"{idx}.pth")
        out = torch.load(file_name)
        out['evaluation'] = evaluator.eval_response(ins, out['response'])
        torch.save(out, file_name)

def main(args):
    dataset = build_dataset(args.dataset)
    data_id = dataset.sample(args.num_samples)

    chunk_idx = get_chunk(data_id, args.num_chunks, args.chunk_idx)
    data, extra_keys = dataset.get_data(chunk_idx)

    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)
    
    get_model_output(args, chunk_idx, data, extra_keys)
    
    eval_model_output(args, chunk_idx, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Evaluate a LVLM model')
    parser.add_argument("--model_name", default="LLaVA-13B")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dataset", default="SEED_2")
    parser.add_argument("--store_path", type=str, default="./output/LLaVA-13B/hidden_states/")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only_ans", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    
    main(parser.parse_args())