import importlib

if importlib.util.find_spec("transformers") is not None:
    from transformers import AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

    class AutoTokenizerWrapper(PreTrainedTokenizer):
        def __new__(cls, *args, **kwargs):
            return AutoTokenizer.from_pretrained(*args, **kwargs)

else:
    raise ModuleNotFoundError("Transformers must be loaded")
