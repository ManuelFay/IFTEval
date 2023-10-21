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

# Define evaluators
lm_evaluator = LMEvaluator(model="gpt-4", cache_path="./outputs/lm_cache/")
lm_evaluator2 = LMEvaluator(model="gpt-3.5-turbo", cache_path="./outputs/lm_cache_gpt3/", max_req_per_s=10)
standard_evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
bertscore_evaluator = BertScoreEvaluator()
sbertscore_evaluator = SBertScoreEvaluator()
rewardmodel_evaluator = RewardModelEvaluator()

datasets = ["mnli", "qnli", "stsb", "sst2", "conll", "xsum", "squadv2"]
# regex = re.compile('fractial_[a-zA-Z0-9]+_\d+_predictions.jsonl')
regex = re.compile("e2_[a-zA-Z0-9]+_\d+_\d_predictions.jsonl")
LM_SAMPLE_SCORING = 150

# BASE_DIR = "outputs/results/results_2804/"
BASE_DIR = "outputs/results/results/"
results = []
l1 = os.listdir(BASE_DIR)
l1 = sorted([x for x in l1 if regex.match(x) and x.split("_")[1] in datasets])

for file in tqdm(l1):
    r = standard_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output")
    r.update(
        CustomEvaluator(custom_key="custom_score", automatic_routing=file).compute_from_jsonl(
            BASE_DIR + file, "prediction", "output"
        )
    )
    r.update(bertscore_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output"))
    r.update(sbertscore_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output"))
    r.update(rewardmodel_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output"))
    if LM_SAMPLE_SCORING > 0:
        lm_res = lm_evaluator.compute_from_jsonl(BASE_DIR + file, "prediction", "output", first_n=LM_SAMPLE_SCORING)
        lm_res2 = lm_evaluator2.compute_from_jsonl(BASE_DIR + file, "prediction", "output", first_n=LM_SAMPLE_SCORING)
        # del lm_res["responses"]
        # del lm_res2["responses"]
        r.update(lm_res)
        r.update(lm_res2)

    print(
        file,
        r["custom_score"],
        r["bertscore"],
        r["sbertscore"],
        r["reward_model_score"],
        r["gpt-4_lm_score"],
        r["gpt-3.5-turbo_lm_score"],
    )
    r["model"] = file.split("_")[1]
    r["num_samples"] = file.split("_")[2]  # .split(".")[0] --> for old format
    r["run_id"] = file.split("_")[3]
    r["file"] = file
    results.append(r)

out_name = "outputs/results_exp1b_standard_datasets.csv"
if os.path.exists(out_name):
    import datetime

    out_name = f"outputs/results_exp1b_standard_datasets_{datetime.date.today()}.csv"
df = pd.DataFrame(results).sort_values(["model", "num_samples"]).to_csv(out_name, index=False)
