import os
import re

import pandas as pd
from tqdm import tqdm

from evaluation.custom_evaluator import CustomEvaluator
from evaluation.standard_evaluator import StandardEvaluator

results = []


regex = re.compile("alpaca_\d+_cat_predictions.jsonl")
standard_evaluator = StandardEvaluator(["exact_match"])
custom_evaluator = None

results = []
cat = ["mnli", "qnli", "stsb", "sst2", "conll", "xsum", "squadv2"]
PATH = """outputs/results/results/results_{cat}/"""

for c in cat:
    for file in tqdm(os.listdir(PATH.format(cat=c))):
        if regex.match(file):
            print(file)
            custom_evaluator = CustomEvaluator(custom_key="custom_score", automatic_routing=c)
            r = standard_evaluator.compute_from_jsonl(PATH.format(cat=c) + file, "prediction", "output")
            r["custom_score"] = custom_evaluator.compute_from_jsonl(PATH.format(cat=c) + file, "prediction", "output")[
                "custom_score"
            ]
            print(file, r["custom_score"])
            r["test_data"] = c
            r["model"] = file.split("_")[0]
            r["num_bins"] = file.split("_")[1]
            r["run_number"] = 0  # file.split("_")[2]
            r["file"] = file
            results.append(r)

df = (
    pd.DataFrame(results)
    .sort_values(["model", "num_bins"])
    .to_csv("outputs/results_exp2_ins_diversity.csv", index=False)
)
