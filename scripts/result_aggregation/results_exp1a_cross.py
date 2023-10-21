import json
import os

import pandas as pd
from tqdm import tqdm

from evaluation.bertscore_evaluator import BertScoreEvaluator
from evaluation.custom_evaluator import CustomEvaluator
from evaluation.lm_evaluator import LMEvaluator
from evaluation.rewardmodel_evaluator import RewardModelEvaluator
from evaluation.sbertscore_evaluator import SBertScoreEvaluator
from evaluation.standard_evaluator import StandardEvaluator

BASE_DIR = "./outputs/results_cross/"
REF_PATH = "./outputs/results_cross/fractial_test_set.json"

standard_evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
lm_evaluator = LMEvaluator(model="gpt-4", cache_path="./outputs/lm_cache/")
lm_evaluator2 = LMEvaluator(model="gpt-3.5-turbo", cache_path="./outputs/lm_cache_gpt3/", max_req_per_s=10)

evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
bertscore_evaluator = BertScoreEvaluator()
sbertscore_evaluator = SBertScoreEvaluator()
rewardmodel_evaluator = RewardModelEvaluator()
# custom_evaluator = CustomEvaluator(custom_key="custom_score", custom_fn=lambda x, y: y.lower() in set(x.lower().replace(".", "").replace(",", "").split()))

with open(REF_PATH, "r") as f:
    ref = json.loads(f.read())

results = []
for file in tqdm(sorted(os.listdir(BASE_DIR))):
    if file.endswith(".jsonl") and file.startswith("results"):
        with open(BASE_DIR + file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        for i, (key, values) in zip(range(0, 18 * 20, 20), ref.items()):
            batch = data[i : i + 20]

            assert ref[key][0]["input"] == batch[0]["input"]
            assert ref[key][19]["input"] == batch[19]["input"]
            assert ref[key][0]["output"] == batch[0]["output"]
            assert ref[key][19]["output"] == batch[19]["output"]

            predictions = [d["prediction"] for d in batch]
            references = [d["output"] for d in batch]
            prompts = [
                ("Instruction:\n" + d["instruction"] + "\nInput:\n" + d["input"])
                if d["input"] != ""
                else ("Instruction:\n" + d["instruction"])
                for d in batch
            ]

            r = {}
            r_std = standard_evaluator._compute(predictions, references, prompts)
            r.update(bertscore_evaluator._compute(predictions, references, prompts))
            # r_custom = custom_evaluator._compute(predictions, references, prompts)
            r_lm = lm_evaluator._compute(predictions, references, prompts)
            r.update(lm_evaluator2._compute(predictions, references, prompts))
            r.update(sbertscore_evaluator._compute(predictions, references, prompts))
            r.update(rewardmodel_evaluator._compute(predictions, references, prompts))
            r["model"] = file.split("_")[2]
            r["num_samples"] = file.split("_")[3].split(".")[0]
            r["file"] = file
            r["target_class"] = key
            r.update(r_std)
            # r.update(r_custom)
            r.update(r_lm)
            results.append(r)

        pd.DataFrame(results).sort_values(["model", "target_class", "num_samples"]).to_csv(
            "outputs/results_cross_tmp.csv", index=False
        )

out_name = "outputs/results_exp1a_cross.csv"
if os.path.exists(out_name):
    import datetime

    out_name = f"outputs/results_exp1a_cross_{datetime.date.today()}.csv"

df = pd.DataFrame(results).sort_values(["model", "target_class", "num_samples"]).to_csv(out_name, index=False)
