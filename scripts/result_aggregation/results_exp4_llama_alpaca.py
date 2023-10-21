import os
import re

import pandas as pd
from tqdm import tqdm

from evaluation.bertscore_evaluator import BertScoreEvaluator
from evaluation.custom_evaluator import CustomEvaluator
from evaluation.lm_evaluator import LMEvaluator
from evaluation.rewardmodel_evaluator import RewardModelEvaluator
from evaluation.sbertscore_evaluator import SBertScoreEvaluator
from evaluation.standard_evaluator import StandardEvaluator

BASE_DIR = "outputs/results/results/"
results = []


regex = re.compile("f(alpaca_|llama_).*jsonl")
lm_evaluator = LMEvaluator(model="gpt-4", cache_path="./outputs/lm_cache/")
# lm_evaluator2 = LMEvaluator(model="gpt-3.5-turbo", cache_path="./outputs/lm_cache_gpt3/", max_req_per_s=10)
standard_evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
# bertscore_evaluator = BertScoreEvaluator()
# sbertscore_evaluator = SBertScoreEvaluator()
# rewardmodel_evaluator = RewardModelEvaluator()

custom_evaluator = None

FIRST_N = 100


files = sorted([file for file in os.listdir(BASE_DIR) if regex.match(file)])
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
        custom_evaluator = CustomEvaluator(custom_key="custom_score", automatic_routing=file)
        r = standard_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output", first_n=FIRST_N)
        # r.update(bertscore_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output"))
        r.update(custom_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output", first_n=FIRST_N))
        # r.update(lm_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output", first_n=FIRST_N))

        print(file, r["custom_score"])
        r["model"] = file.split("_")[0]
        r["task"] = file.split("_")[1]
        r["num_samples"] = file.split("_")[2]
        r["run_number"] = file.split("_")[3]
        r["file"] = file
        results.append(r)

df = (
    pd.DataFrame(results)
    .sort_values(["model", "num_samples"])
    .to_csv("outputs/results_exp4_alpaca_llama.csv", index=False)
)
