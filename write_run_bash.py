# Description: Write bash script to run evaluate_lvlm.py
from glob import glob

from tqdm import tqdm
from utils.config import MODEL_LIST, TASK_INFO

prefix = """#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

"""

parallel_tmp = """for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} %s
done

wait

"""

exp_tmp = """python -m evaluate_lvlm \\%s
        --model_name %s \\
        --model_path %s \\
        --dataset %s \\
        --store_path ./output/%s/%s_HS/ \\
        --num_chunks $CHUNKS \\
        --chunk_idx $IDX \\
        --temperature 0.0 \\
        --num_beams 1 \\
        --only_ans &
"""

SUB_SAMPLING_TEMP = "\n        --num_samples %s \\"

with open("./scripts/new_experiments.sh", "w") as f, open("./scripts/gpt_experiments.sh", "w") as f2:
    f.write(prefix)
    print("Number of models: ", len(MODEL_LIST))
    print("Number of tasks: ", len(TASK_INFO))
    
    for model_meta_info in tqdm(MODEL_LIST, position=0, leave=True):
        model_name = model_meta_info["model_name"]
        model_path = model_meta_info["model_path"]
        store_model_path = model_meta_info["store_model_path"]

        for task in tqdm(TASK_INFO, position=1, leave=False):
            name, store_path = task["dataset"], task["dataset"]

            model_out_path = "./output/%s/%s_HS/*" % (store_model_path, store_path)
            num_samples = len(glob(model_out_path))

            if model_name in ["Gemini", "GPT4"] and task["num_samples"] > 100 and task["sub_sampling"] is None:
                sub_sampling = 100
                task_num_samples = 100
            else:
                sub_sampling = task["sub_sampling"]
                task_num_samples = task["num_samples"]

            if sub_sampling is None:
                sub_sampling_str = ""
            else:
                sub_sampling_str = SUB_SAMPLING_TEMP % sub_sampling
                task_num_samples = sub_sampling

            if num_samples < task_num_samples:
                this_exp = exp_tmp % (sub_sampling_str, model_name, model_path, name, store_model_path, store_path)
                if model_name in ["Gemini", "GPT4"]:
                    this_exp = this_exp.replace("$CHUNKS", "1").replace("$IDX", "0")
                    f2.write(this_exp + "wait\n\n")
                else:
                    f.write(parallel_tmp % this_exp)
