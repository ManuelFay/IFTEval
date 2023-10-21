from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.base_evaluator import BaseEvaluator


class RewardModelEvaluator(BaseEvaluator):
    def __init__(self, model_type="OpenAssistant/reward-model-deberta-v3-large-v2", batch_size=1, **kwargs):
        """Initialize the evaluator with a list of metrics to compute

        clf_metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
        qa_metrics: List[str] = ["squad"]
        summarization_metrics: List[str] = ["rouge1", "rouge2", "rougeL", "bleu"]
        translation_metrics: List[str] = ["bleu"]
        """
        super().__init__(**kwargs)
        self.name = "reward_model_evaluator_{}".format(model_type)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = AutoModelForSequenceClassification.from_pretrained(model_type).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model_type = model_type

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]
        prompts = args[2]

        if self.batch_size > 1:
            raise NotImplementedError
        else:
            results, results_ref = [], []
            for prediction, reference, prompt in zip(predictions, references, prompts):
                res = self.get_cache(self.name, prompt + prediction)
                if res is None:
                    with torch.no_grad():
                        inputs = self.tokenizer(prompt, prediction, return_tensors="pt").to(self.device)
                        res = self.scorer(**inputs).logits[0].cpu().detach()
                    self.set_cache(self.name, prompt + prediction, res)
                results.append(res)
                res2 = self.get_cache(self.name, prompt + reference)
                if res2 is None:
                    with torch.no_grad():
                        inputs = self.tokenizer(prompt, reference, return_tensors="pt").to(self.device)
                        res2 = self.scorer(**inputs).logits[0].cpu().detach()
                    self.set_cache(self.name, prompt + reference, res2)
                results_ref.append(res2)
            results = torch.stack(results)
            results_ref = torch.stack(results_ref)
            softmaxed = torch.softmax(torch.stack([results, results_ref]), dim=0)[0]
        response = {
            "reward_model_results": [results.cpu().tolist(), softmaxed.cpu().tolist()],
            "reward_model_score": results.mean().item(),
            "softmaxed_reward_model_score": softmaxed.mean().item(),
        }
        return response


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):
    evaluator = RewardModelEvaluator(**kwargs)
    print(evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key))


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
