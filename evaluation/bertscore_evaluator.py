from functools import lru_cache
from typing import List

import torch
from bert_score import BERTScorer

from evaluation.base_evaluator import BaseEvaluator


class BertScoreEvaluator(BaseEvaluator):
    def __init__(self, model_type="microsoft/deberta-base-mnli", lang=None, batch_size=1, **kwargs):
        """Initialize the evaluator with a list of metrics to compute

        clf_metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
        qa_metrics: List[str] = ["squad"]
        summarization_metrics: List[str] = ["rouge1", "rouge2", "rougeL", "bleu"]
        translation_metrics: List[str] = ["bleu"]
        """
        super().__init__(**kwargs)
        self.name = "bertscore_evaluator_{}".format(model_type)
        self.batch_size = batch_size
        self.bertscorer = BERTScorer(
            model_type=model_type, lang=lang, rescale_with_baseline=lang is not None, batch_size=batch_size
        )
        self.model_type = model_type
        self.lang = lang

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]

        if self.batch_size > 1:
            results = self.bertscorer.score(predictions, references)[2]
        else:
            results = []
            for prediction, reference in zip(predictions, references):
                res = self.get_cache(self.name, prediction + reference)
                if res is None:
                    res = self.bertscorer.score([prediction], [reference])[2][0]
                    self.set_cache(self.name, prediction + reference, res)
                results.append(res)
            results = torch.stack(results)
        response = {"bertscore_results": results.cpu().tolist(), "bertscore": results.mean().item()}
        return response


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):
    evaluator = BertScoreEvaluator(**kwargs)
    print(evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key))


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
