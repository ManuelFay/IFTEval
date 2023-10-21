import os
import re

import pandas as pd
from tqdm import tqdm

from evaluation.custom_evaluator import CustomEvaluator
from evaluation.lm_evaluator import LMEvaluator
from evaluation.standard_evaluator import StandardEvaluator

BASE_DIR = "outputs/results/results/"
results = []

cats = ["classify", "create", "sst2"]  # , "xsum"]

regex = re.compile("alpaca_\d+_\d_predictions.jsonl")
standard_evaluator = StandardEvaluator(["exact_match", "rouge"])
lm_evaluator = LMEvaluator(model="gpt-4", cache_path="./outputs/lm_cache/")
FIRST_N = 100
custom_evaluator = None

for cat in cats:
    path = BASE_DIR + f"exp3_{cat}/"
    files = sorted([file for file in os.listdir(path) if regex.match(file)])
    seen = {}
    pruned_files = []
    for file in files:
        key = "".join(file.split("_")[:3])
        if key in seen.keys():
            num = seen[key]
            print(key, num)
            if num > 4:
                continue
        else:
            seen[key] = 0
        seen[key] += 1
        pruned_files.append(file)

    for file in tqdm(pruned_files):
        if regex.match(file):
            print(file)
            custom_evaluator = CustomEvaluator(custom_key="custom_score", automatic_routing=cat)
            r = standard_evaluator.compute_from_jsonl(path + file, "prediction", "output", first_n=FIRST_N)
            r.update(custom_evaluator.compute_from_jsonl(path + file, "prediction", "output", first_n=FIRST_N))
            if cat in ["classify", "create"]:
                r.update(lm_evaluator.compute_from_jsonl(path + file, "prediction", "output", first_n=FIRST_N))
            print(file, r["custom_score"])
            r["model"] = file.split("_")[0]
            r["num_samples"] = file.split("_")[1]
            r["run_number"] = file.split("_")[2]
            r["file"] = file
            r["task"] = cat
            results.append(r)

df = pd.DataFrame(results).sort_values(["model", "num_samples"]).to_csv("outputs/results_exp3_scaling.csv", index=False)
