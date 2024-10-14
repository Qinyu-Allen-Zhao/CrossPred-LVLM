import math
import subprocess
import json

import torch
from glob import glob
import numpy as np
from tqdm import tqdm
from scipy import linalg
from evaluate import load
from sklearn.metrics import precision_score, recall_score, f1_score


def shell_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()  
    p_status = p.wait()

    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True) # only difference


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size].tolist() for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_jsonl(file, num=None):
    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            data.append(json.loads(line))
            
            if num and i == num:
                break

    return data

# def read_data(model, dataset, num_tasks_expected, num_samples_expected):
#     file_num = len(glob(f"./output/{model}/{dataset}_HS/*.pth"))
#     files = [f"./output/{model}/{dataset}_HS/{i}.pth" for i in range(file_num)]
#     data = [torch.load(file) for file in tqdm(files)]

#     task_labels = [ins['category'] for ins in data]
#     task_types = []
#     task_data = {}

#     for task in np.unique(task_labels):
#         sub_data = [ins for ins in data if ins['category'] == task]
#         if len(sub_data) < 30:
#             print(f"Task: {task} | Insufficient data")
#             continue
        
#         task_types.append(f"{task} ({dataset})")
        
#         # MSP
#         metric = [softmax(ins['logits'][[0]]).max() for ins in sub_data]

#         # Max Logit
#         # metric = [ins['logits'][0].max() for ins in sub_data]

#         # Top1 - top2
#         # metric = []
#         # for ins in sub_data:
#         #     sp = np.sort(softmax(ins['logits'][[0]])[0])
#         #     metric.append(sp[-1] - sp[-2])

#         # # 1/N * sum(softmax)
#         # metric = []
#         # for ins in sub_data:
#         #     sp = softmax(ins['logits'][:-1]).max(axis=1)
#         #     metric.append(np.mean(np.log(sp)))

#         # Max of ABCD
#         # metric = [softmax(ins['logits'][:, [319, 350, 315, 360]])[0].max() for ins in sub_data]

#         acc = np.mean([ins['evaluation'] for ins in sub_data]) * 100
#         task_data[f"{task} ({dataset})"] = {
#             "acc": acc,
#             "metric": metric
#         }
        
#     # Total Acc
#     tot_acc = np.mean([ins['evaluation'] for ins in data]) * 100
#     print(f"Total Acc: {tot_acc:.2f}%, Num Tasks: {len(task_types)}, Num Samples: {len(data)}")

#     assert len(task_types) == num_tasks_expected, f"Expected {num_tasks_expected} tasks, got {len(task_types)}"
#     assert len(data) == num_samples_expected, f"Expected {num_samples_expected} samples, got {len(data)}"

#     return task_data, task_types

def read_data(model, dataset, output_path="./output", eval_gather="avg"):
    files = glob(f"{output_path}/{model}/{dataset}_HS/*.pth")
    data = {}

    print("Reading results ...")
    all_res = []
    for file in tqdm(files):
        ins = torch.load(file)
        if ins['response'] == "<BLOCKED>":
            print(ins)
            continue
        
        task = f"{ins['category']} ({dataset})"
        if task not in data:
            data[task] = []
        data[task].append(ins)
        all_res.append(ins['evaluation'])

    task_labels = list(data.keys())
    task_data = {}

    print("Sub Tasks ...")
    for task in task_labels:
        sub_data = data[task]
        if len(sub_data) < 28:
            print(f"Task: {task} | Insufficient data")
            continue
        if eval_gather == "avg":
            perf = {"acc": np.mean([ins['evaluation'] for ins in sub_data])}
        elif eval_gather == "true_false":
            pred = [ins['evaluation'] if ins['label'].lower()=='yes' else not ins['evaluation'] for ins in sub_data]
            gt = [ins['label'].lower()=='yes' for ins in sub_data]

            perf = {
                "acc": np.mean([ins['evaluation'] for ins in sub_data]), # accuracy
                "prec": precision_score(gt, pred),
                "rec": recall_score(gt, pred),
                "f1": f1_score(gt, pred)
            }
        elif eval_gather == "hemm":
            bertscore = load("bertscore")
            predictions = [ins['response'] for ins in sub_data]
            references = [ins['label'] for ins in sub_data]
            results = bertscore.compute(predictions=predictions, references=references, lang="en")

            perf = {"bart": np.mean([ins['evaluation'] for ins in sub_data]),
                    "bert": np.mean(results['f1'])}
        else:
            raise ValueError(f"Unknown eval_gather: {eval_gather}")

        task_data[task] = perf
    
    num_good_tasks = len(task_data.keys())
    avg_perf = np.mean(all_res)
    print(f"Avg Perf: {avg_perf:.2f}, Num Tasks: {num_good_tasks}, Num Samples: {len(all_res)}")

    return task_data


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
