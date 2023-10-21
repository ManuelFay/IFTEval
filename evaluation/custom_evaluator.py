import re
import string
from collections import Counter
from functools import lru_cache
from typing import Callable, Optional

import evaluate
import numpy as np

from evaluation.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    def __init__(
        self, custom_key=None, custom_fn: Optional[Callable] = None, automatic_routing: Optional[str] = None, **kwargs
    ):
        """Initialize the evaluator with a list of metrics to compute
        custom_fns: List[Callable] with each function taking in predictions and references and returning a float score
        """
        super().__init__(**kwargs)
        self.custom_key = custom_key if custom_key is not None else "custom_score"
        self.custom_fn = custom_fn
        self.name = "custom_evaluator"
        self.rouge_eval = evaluate.combine(["rouge", "exact_match"])
        self.squad_eval = evaluate.combine(["squad"])
        self.custom_fn = self.router(automatic_routing) if automatic_routing is not None else self.custom_fn

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def mnli_custom_evaluator(x, y):
        options = ["contradiction", "entailment", "neutral"]
        lowest_index = 10000, ""
        for option in options:
            if option[:7] in x.lower():
                if x.lower().index(option[:7]) < lowest_index[0]:
                    lowest_index = x.lower().index(option[:7]), option
        if lowest_index[1] == "":
            return False
        return lowest_index[1] == y

    @staticmethod
    def qnli_custom_evaluator(x, y):
        if ("unanswerable" in x.lower()) or ("does not" in x.lower()):
            return "unanswerable" == y
        return "answerable" == y

    @staticmethod
    @lru_cache(maxsize=10000)
    def sst2_custom_evaluator(x, y):
        if "negative" in x.lower():
            if "positive" in x.lower():
                # Undecided
                return False
            return "negative" == y
        return "positive" == y

    @staticmethod
    def stsb_custom_evaluator(x, y):
        # match a digit
        x = re.findall(r"[-+]?\d*\.\d+|\d+", x)
        if len(x) == 0:
            return 0
        else:
            x = float(x[0])
        y = float(y)
        # print(x, y)
        # return 1 - min(5, max(0, abs(x-y)))/5
        return x == y

    @staticmethod
    def conll_custom_evaluator(x, y):
        try:
            return eval(x.replace("```", "")) == eval(y)
        except:
            return False
        # return rouge_eval.compute(predictions=[str(x)], references=[str(y)])["rouge1"]

    def xsum_custom_evaluator(self, x, y):
        return self.rouge_eval.compute(predictions=[x], references=[y])["rouge2"]

    def squad_custom_evaluator(self, x, y):
        return self.f1_score(x, y)

    def router(self, file):
        if "mnli" in file:
            return self.mnli_custom_evaluator
        elif "qnli" in file:
            return self.qnli_custom_evaluator
        elif "stsb" in file:
            return self.stsb_custom_evaluator
        elif "sst2" in file:
            return self.sst2_custom_evaluator
        elif "conll" in file:
            return self.conll_custom_evaluator
        elif "xsum" in file:
            return self.xsum_custom_evaluator
        elif "squad" in file:
            return self.squad_custom_evaluator
        return lambda x, y: -1

    def _compute(self, *args, **kwargs):
        predictions = args[0]
        references = args[1]
        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores += [self.custom_fn(pred, ref)]

        return {
            self.custom_key: np.nanmean(scores),
            "responses": [
                {"prediction": pred, "ref": ref, self.custom_key: sc}
                for pred, ref, sc in zip(predictions, references, scores)
            ],
        }


def run_cli(predictions_file: str, predictions_key: str, reference_key: str, **kwargs):

    evaluator = CustomEvaluator(
        custom_key=kwargs.get("custom_key", "custom_score"),
        custom_fn=kwargs.get(
            "custom_fn", lambda x, y: y.lower() in set(x.lower().replace(".", "").replace(",", "").split())
        ),
        **kwargs,
    )
    scores = evaluator.compute_from_jsonl(predictions_file, predictions_key, reference_key)
    print(f"Mean score prediction: {scores['custom_score']}")


if __name__ == "__main__":
    import fire

    fire.Fire(run_cli)
