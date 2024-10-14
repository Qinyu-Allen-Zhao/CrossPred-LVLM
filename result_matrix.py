import re
import os
import json
from glob import glob
from tqdm import tqdm
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from utils.func import calculate_frechet_distance, read_data, softmax
from utils.visualize import draw_task, plot_two_rel
from utils.config import MODEL_LIST, TASK_INFO, ALL_DATASETS
from dataset import build_dataset

output_path = "./output"

# Validate the results

gpt_samples = {
    "SEED_2": 2606,
    "MME": 1000,
    "MMBench_CN": 1994,
    "MMBench_EN": 1994,
    "MMMU": 900,
    "CMMMU": 573,
    "ScienceQA": 1467,
    "CVBench": 400
}


bar = tqdm(MODEL_LIST)
for model in bar:
    bar.set_description(model["model_path"])
    for task in TASK_INFO:
        file_list = glob(f"{output_path}/{model['store_model_path']}/{task['dataset']}_HS/*.pth")

        if model["model_name"] in ["Gemini", "GPT4"]:
            if task["dataset"] in gpt_samples:
                assert len(file_list) == gpt_samples[task["dataset"]], f"Dataset: {task['dataset']}, Model: {model['model_name']} Expected: {gpt_samples[task['dataset']]} Got: {len(file_list)}"
            else:
                assert len(file_list) == task["sub_sampling"], f"Dataset: {task['dataset']}, Model: {model['model_name']} Expected: {task['sub_sampling']} Got: {len(file_list)}"
        else:
            if task["sub_sampling"] is not None:
                assert len(file_list) == task["sub_sampling"]
            else:   
                assert len(file_list) == task["num_samples"]

        for file in file_list:
            ins = torch.load(file)
            ins2 = torch.load(file.replace(model['store_model_path'], "BLIP2-opt-2.7B"))
            for key in ["question", "label"]:
                if isinstance(ins[key], list):
                    ins[key] = sorted(ins[key])
                    ins2[key] = sorted(ins2[key])

                if ins[key] != ins2[key]:
                    print(f"Key: {key} not equal for {file}")
                    print(ins)
                    print(ins2)
                    raise ValueError("Key not equal")


# Main result summary
with open(f"{output_path}/result_summary.json", "r") as f:
    result_summary = json.load(f)

for model_info in MODEL_LIST:
    model_name = model_info['store_model_path']
    print(f"Processing {model_name}")

    task_data_all = result_summary.get(model_name, {})
    exisiting_tasks = set(re.findall(r'\((.*?)\)', key)[0] for key in task_data_all.keys())

    for task_i in TASK_INFO:
        if task_i['dataset'] in exisiting_tasks:
            continue
        print("-- Update task: ", task_i['dataset'])
        task_data = read_data(model_name, task_i['dataset'], output_path, "avg")
        task_data_all.update(task_data)

    result_summary.update({model_name: task_data_all})

with open(f"{output_path}/result_summary.json", "w") as f:
    json.dump(result_summary, f, indent=4)


# Dataset representation
for dataset in tqdm(ALL_DATASETS, position=1):
    task_name, bench_name = re.match(r"(.*) \((.*)\)", dataset).groups()
    output_file = f"{output_path}/dataset_representation/{task_name}_{bench_name}.pkl"
    if os.path.exists(output_file):
        continue
            
    files = glob(f"{output_path}/LLaVA-7B/{bench_name}_HS/*.pth")
    data = []
    for file in tqdm(files, position=0):
        ins = torch.load(file)
        if ins["category"] != task_name:
            continue
        for key, value in ins.items():
            if isinstance(value, torch.Tensor):
                ins[key] = value.cpu().numpy()
                
        data.append(ins)

    print(len(data))
    joblib.dump(data, output_file)


# Dataset representation from CLIP
for dataset in tqdm(ALL_DATASETS, position=1):
    task_name, bench_name = re.match(r"(.*) \((.*)\)", dataset).groups()
    output_file = f"{output_path}/dataset_representation_clip/{task_name}_{bench_name}.pkl"
    if os.path.exists(output_file):
        continue
            
    files = glob(f"{output_path}/CLIP_embed/{bench_name}_HS/*.pth")
    data = []
    for file in tqdm(files, position=0):
        ins = torch.load(file)
        if ins["category"] != task_name:
            continue
        for key, value in ins.items():
            if isinstance(value, torch.Tensor):
                ins[key] = value.cpu().numpy()
                
        data.append(ins)

    print(len(data))
    joblib.dump(data, output_file)


# All result tensor
all_result_summary_file = f"{output_path}/all_result_summary.json"
if not os.path.exists(all_result_summary_file):
    with open(all_result_summary_file, "w") as f:
        json.dump({}, f, indent=4)

with open(all_result_summary_file, "r") as f:
    result_summary = json.load(f)

for model_info in MODEL_LIST:
    model_name = model_info['store_model_path']
    print(f"Processing {model_name}")

    task_data_all = result_summary.get(model_name, {})
    exisiting_tasks = set(re.findall(r'\((.*?)\)', key)[0] for key in task_data_all.keys())

    for task_i in TASK_INFO:
        if task_i['dataset'] in exisiting_tasks:
            continue

        print("-- Update task: ", task_i['dataset'])
        task_data = read_data(model_name, task_i['dataset'], output_path, task_i["evaluation"])
        task_data_all.update(task_data)

    result_summary.update({model_name: task_data_all})

with open(all_result_summary_file, "w") as f:
    json.dump(result_summary, f, indent=4)