import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from evaluation.bertscore_evaluator import BertScoreEvaluator
from evaluation.custom_evaluator import CustomEvaluator
from evaluation.lm_evaluator import LMEvaluator
from evaluation.rewardmodel_evaluator import RewardModelEvaluator
from evaluation.sbertscore_evaluator import SBertScoreEvaluator
from evaluation.standard_evaluator import StandardEvaluator

standard_evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
lm_evaluator = LMEvaluator(model="gpt-4", cache_path="../results/outputs/lm_cache/")
lm_evaluator2 = LMEvaluator(model="gpt-3.5-turbo", cache_path="../results/outputs/lm_cache_gpt3/", max_req_per_s=10)

evaluator = StandardEvaluator(metrics=["rouge", "exact_match"])
bertscore_evaluator = BertScoreEvaluator()
sbertscore_evaluator = SBertScoreEvaluator()
# rewardmodel_evaluator = RewardModelEvaluator()


df = load_dataset("manu/REALSumm")["train"].to_pandas()

# For quicker results
df = df[df.model.isin(["t5_out_11B.txt", "bart_out.txt", "unilm_out_v2.txt"])]

results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    predictions = [row.model_summary]
    references = [row.ref_summary]
    prompts = [
        ("Instruction:\n" + "Summarize the following news article in a few sentences." + "\nInput:\n" + row.source)
    ]

    r = {}
    r.update(standard_evaluator._compute(predictions, references, prompts))
    r.update(bertscore_evaluator._compute(predictions, references, prompts))
    # r_custom = custom_evaluator._compute(predictions, references, prompts)
    r_lm = lm_evaluator._compute(predictions, references, prompts)
    r.update(r_lm)
    # r.update(lm_evaluator2._compute(predictions, references, prompts))
    r.update(sbertscore_evaluator._compute(predictions, references, prompts))
    # r.update(rewardmodel_evaluator._compute(predictions, references, prompts))

    results.append(r)
df2 = pd.DataFrame(results)
print(len(df), len(df2))

df = pd.concat((df.reset_index(drop=True), df2.reset_index(drop=True)), axis=1)
print(len(df))
df.to_csv("outputs/results_exp0_sum.csv", index=False)
