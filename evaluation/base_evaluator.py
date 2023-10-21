import json

from diskcache import Cache


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        """Initialize the evaluator with a list of metrics to compute

        clf_metrics: List[str] = ["accuracy", "f1", "precision", "recall"]
        qa_metrics: List[str] = ["squad"]
        summarization_metrics: List[str] = ["rouge1", "rouge2", "rougeL", "bleu"]
        translation_metrics: List[str] = ["bleu"]
        """
        self.name = "base"
        cache_path = kwargs.get("cache_path", "~/.cache/evaluator")
        self.cache = Cache(cache_path) if cache_path else None

    @staticmethod
    def get_cache_key(model_name, text):
        return f"{model_name}_{text}"

    def set_cache(self, model_name, text, response):
        if self.cache is not None:
            cache_key = self.get_cache_key(model_name, text)
            self.cache.set(cache_key, response)

    def get_cache(self, model_name, text):
        if self.cache is not None:
            cache_key = self.get_cache_key(model_name, text)
            return self.cache.get(cache_key)
        return None

    def _compute(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Compute the metrics on the given predictions and references
        predictions: List[str]
        references: List[str]
        """
        return self._compute(*args, **kwargs)

    def compute_from_jsonl(
        self,
        predictions_file: str,
        predictions_key: str,
        reference_key: str,
        first_n: int = None,
        save_to_file: bool = False,
        **kwargs,
    ):
        """Compute the metrics on the given predictions and references
        predictions_file: str
        references_file: str
        """
        assert predictions_file.endswith(".jsonl")
        predictions = []
        references = []
        prompts = []
        with open(predictions_file) as f:
            for line in f:
                if first_n and len(predictions) >= first_n:
                    break
                data = json.loads(line)
                predictions.append(data[predictions_key])
                references.append(data[reference_key])
                if data["input"] != "":
                    prompts.append("Instruction:\n" + data["instruction"] + "\nInput:\n" + data["input"])
                else:
                    prompts.append("Instruction:\n" + data["instruction"])
        scores = self._compute(predictions, references, prompts, **kwargs)

        if save_to_file:
            with open(predictions_file.replace(".jsonl", f"{self.name}_scores.json"), "w") as f:
                f.write(json.dumps(scores, indent=4))
        return scores
