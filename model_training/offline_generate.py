import json
import os
import sys

import fire
import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    data_path: str,
    output_dir: str,
    load_8bit: bool = True,
    batch_size: int = 4,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "org-name/your-model-name",
    lora_revision: str = "main",
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    **kwargs,
):
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(lora_weights, use_auth_token=True, revision=lora_revision)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            use_auth_token=True,
            revision=lora_revision,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            use_auth_token=True,
            device_map={"": device},
            torch_dtype=torch.float16,
            revision=lora_revision,
        )
    else:
        raise NotImplementedError("Only cuda and mps are supported at this time.")
        # model = LlamaForCausalLM.from_pretrained(
        #     base_model, device_map={"": device}, low_cpu_mem_usage=True
        # )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     use_auth_token=True,
        #     revision=lora_revision,
        #     device_map={"": device},
        # )

    print(
        f"Training Alpaca-LoRA model {lora_weights} with params:\n"
        f"tokenizer eos token: {tokenizer.eos_token_id}:{tokenizer.eos_token}\n"
        f"tokenizer pad token: {tokenizer.pad_token_id}:{tokenizer.pad_token}\n"
        f"tokenizer bos token: {tokenizer.bos_token_id}:{tokenizer.bos_token}\n"
        f"tokenizer unk token: {tokenizer.unk_token_id}:{tokenizer.unk_token}\n"
        f"model eos token: {model.config.eos_token_id}:{tokenizer.eos_token}\n"
        f"model pad token: {model.config.pad_token_id}:{tokenizer.pad_token}\n"
        f"model bos token: {model.config.bos_token_id}:{tokenizer.bos_token}\n"
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"tokenizer eos token: {tokenizer.eos_token_id}:{tokenizer.eos_token}\n"
        f"tokenizer pad token: {tokenizer.pad_token_id}:{tokenizer.pad_token}\n"
        f"tokenizer bos token: {tokenizer.bos_token_id}:{tokenizer.bos_token}\n"
        f"tokenizer unk token: {tokenizer.unk_token_id}:{tokenizer.unk_token}\n"
        f"model eos token: {model.config.eos_token_id}:{tokenizer.eos_token}\n"
        f"model pad token: {model.config.pad_token_id}:{tokenizer.pad_token}\n"
        f"model bos token: {model.config.bos_token_id}:{tokenizer.bos_token}\n"
    )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    data = load_dataset("json", data_files=data_path)
    data = DataLoader(data["train"], batch_size=batch_size, shuffle=False)

    def evaluate(
        instructions,
        inputs,
        temperature=0.1,
        top_p=0.75,
        length_pen=1.0,
        num_beams=4,
        max_new_tokens=256,
        repetition_penalty=1.0,
        top_k=40,
        max_input_length=512,
        **kwargs,
    ):
        # print(f"Running inference in {'streaming' if stream_output else 'batched'} mode")

        prompts = [prompter.generate_prompt(ins, inp) for ins, inp in zip(instructions, inputs)]
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_input_length)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            # do_sample=True,
            top_p=top_p,
            top_k=top_k,
            length_penalty=length_pen,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )

        outputs = tokenizer.batch_decode(
            generation_output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        formatted_answers = [prompter.get_batch_response(output) for output in outputs]
        return formatted_answers

    generation_params = {
        "temperature": kwargs.get("temperature", 0.1),
        "top_p": kwargs.get("top_p", 0.75),
        "length_pen": kwargs.get("length_pen", 1.0),
        "num_beams": kwargs.get("num_beams", 1),
        "max_new_tokens": kwargs.get("max_new_tokens", 256),
        "top_k": kwargs.get("top_k", 40),
        "top_k": kwargs.get("top_k", 40),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        "device": device,
        "batch_size": batch_size,
        "max_input_length": kwargs.get("max_input_length", 512),
    }

    print(generation_params)

    def sanitize_filename(filename: str) -> str:
        return "".join(c for c in filename if c.isalnum() or c in "._- ").replace("models", "")

    filename = os.path.join(output_dir, f"{sanitize_filename(lora_weights)}_predictions.jsonl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(filename):
        os.remove(filename)

    for batch in tqdm(data):
        print(batch)
        preds = evaluate(batch["instruction"], batch["input"], **generation_params)
        print(preds)

        with open(filename, "a+") as f:
            for ins, inp, ans, pred in zip(batch["instruction"], batch["input"], batch["output"], preds):
                f.write(json.dumps({"instruction": ins, "input": inp, "output": ans, "prediction": pred}) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
