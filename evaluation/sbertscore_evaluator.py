from typing import List

import torch
from sentence_transformers import SentenceTransformer, util

from evaluation.base_evaluator import BaseEvaluator


class SBertScoreEvaluator(BaseEvaluator):
    def __init__(self, model_type="all-MiniLM-L6-v2", batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.name = "sbertscore_evaluator_{}".format(model_type)
        self.batch_size = batch_size
        self.embedder = SentenceTransformer(model_type)
        self.model_type = model_type

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]

        if self.batch_size > 1:
            predictions = self.embedder.encode(predictions)
            references = self.embedder.encode(references)
            results = util.pairwise_cos_sim(predictions, references)

        else:
            results = []
            for prediction, reference in zip(predictions, references):
                res = self.get_cache(self.name, prediction + reference)
                if res is None:
                    prediction = self.embedder.encode([prediction])
                    reference = self.embedder.encode([reference])
                    res = util.pairwise_cos_sim(prediction, reference)
                    self.set_cache(self.name, prediction + reference, res)
                results.append(res)
            results = torch.stack(results)
        return {"sbertscore_results": results.cpu().tolist(), "sbertscore": results.mean().item()}


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):
    evaluator = SBertScoreEvaluator(**kwargs)
    print(evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key))


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
