from typing import List

import evaluate

from evaluation.base_evaluator import BaseEvaluator


class StandardEvaluator(BaseEvaluator):
    def __init__(self, metrics: List[str] = ["rouge", "exact_match"], **kwargs):
        """Initialize the evaluator with a list of metrics to compute

        clf_metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
        qa_metrics: List[str] = ["squad"]
        summarization_metrics: List[str] = ["rouge1", "rouge2", "rougeL", "bleu"]
        translation_metrics: List[str] = ["bleu"]
        """
        super().__init__(**kwargs)
        self.name = "standard_evaluator"
        self.metrics = evaluate.combine(metrics)

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]
        return self.metrics.compute(predictions, references)


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):
    evaluator = StandardEvaluator(**kwargs)
    print(evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key))


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
