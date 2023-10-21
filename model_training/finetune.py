import json
import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # get_peft_model_state_dict,; set_peft_model_state_dict,
    LoraConfig, get_peft_model, prepare_model_for_int8_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)

from utils.prompter import Prompter


def evaluate(
    instructions,
    inputs,
    model,
    tokenizer,
    temperature=0.1,
    top_p=0.75,
    length_pen=1.0,
    num_beams=1,
    max_new_tokens=128,
    repetition_penalty=1.0,
    top_k=40,
    **kwargs,
):

    # print(f"Running inference in {'streaming' if stream_output else 'batched'} mode")
    prompter = Prompter("alpaca_short")

    prompts = [prompter.generate_prompt(ins, inp) for ins, inp in zip(instructions, inputs)]
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True, padding=True, max_length=kwargs.get("max_input_length", 512)
    )
    input_ids = inputs["input_ids"].to(model.device)
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


def train(
    # model/data params
    base_model: str = "huggyllama/llama-7b",  # the only required argument
    train_data_path: Optional[str] = None,
    output_dir: str = "data/lora-alpaca-french",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    predict_batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    random_seed: int = 42,
    val_data_path: Optional[str] = None,
    val_set_size: Optional[int] = None,
    test_set_size: Optional[int] = None,
    test_data_path: Optional[str] = None,
    pred_file_name: Optional[str] = "predictions.jsonl",
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    lora_pretrained_path: Optional[str] = None,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    push_to_hub: bool = False,
    hub_model_id: str = None,
    padding_side: str = "left",
    bf16: bool = False,
    load_in_8bit: bool = False,
    save_whole_model: bool = False,
    **kwargs,
):
    if lora_target_modules is None:
        if "llama" in base_model:  # huggyllama/llama-7b
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        if "falcon" in base_model:  # tiiuae/falcon-7b
            lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        if "bloom" in base_model:  # bigscience/bloom-7b1
            lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        if "opt" in base_model:  # facebook/opt-6.7b
            lora_target_modules = ["q_proj", "v_proj"]
        if "pythia" in base_model:  # EleutherAI/pythia-6.9b
            lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {train_data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"test_data_path: {test_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n,"
            f"padding_side: {padding_side}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print(f"Using DDP with {world_size} GPUs, gradient_accumulation_steps={gradient_accumulation_steps}")

    def create_and_prepare_model(float_dtype):
        compute_dtype = getattr(torch, float_dtype)
        print(f"Using compute_dtype={compute_dtype} for training, and float_dtype={float_dtype}")

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=False,
        # )
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True,
        #                                 bnb_8bit_compute_dtype=compute_dtype)

        if compute_dtype == torch.float16 and not bf16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

        device_map = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            # quantization_config=bnb_config,
            load_in_8bit=load_in_8bit,  # makes it slower
        )

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.bos_token = tokenizer.eos_token

        model.config.bos_token_id = model.config.eos_token_id
        model.config.pad_token_id = 0

        return model, peft_config, tokenizer

    # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #         "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model, peft_config, tokenizer = create_and_prepare_model(float_dtype="bfloat16" if bf16 else "float16")
    model.config.use_cache = False

    # model.config.pad_token_id = 0  # unk. we want this to be different from the eos token
    # model.config.eos_token_id = 2  # eos
    # model.config.bos_token_id = 1  # bos
    # model.config.unk_token_id = 0  # unk
    tokenizer.padding_side = padding_side  # Allow batched inference

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"tokenizer eos token: {tokenizer.eos_token_id}:{tokenizer.eos_token}\n"
            f"tokenizer pad token: {tokenizer.pad_token_id}:{tokenizer.pad_token}\n"
            f"tokenizer bos token: {tokenizer.bos_token_id}:{tokenizer.bos_token}\n"
            f"model eos token: {model.config.eos_token_id}:{tokenizer.eos_token}\n"
            f"model pad token: {model.config.pad_token_id}:{tokenizer.pad_token}\n"
            f"model bos token: {model.config.bos_token_id}:{tokenizer.bos_token}\n"
            f"tokenizer padding side: {tokenizer.padding_side}\n"
            f"tokenizer max length: {tokenizer.model_max_length}\n"
        )

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if load_in_8bit:
        model = prepare_model_for_int8_training(model)  # makes it slower

    if lora_pretrained_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            lora_pretrained_path,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            use_auth_token=True,
        )
        model = model.merge_and_unload()
        model.config.use_cache = True
        model.config.inference_mode = False

    model = get_peft_model(model, peft_config)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    model.config.bos_token_id = model.config.eos_token_id
    model.config.pad_token_id = 0

    train_data = None
    if train_data_path:
        if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
            train_data = load_dataset("json", data_files=train_data_path)
        else:
            train_data = load_dataset(train_data_path)

    val_data = None
    if val_data_path:
        if val_data_path.endswith(".json") or val_data_path.endswith(".jsonl"):
            val_data = load_dataset("json", data_files=val_data_path)
        else:
            val_data = load_dataset(val_data_path)

    test_data = None
    if test_data_path:
        if test_data_path.endswith(".json") or test_data_path.endswith(".jsonl"):
            test_data = load_dataset("json", data_files=test_data_path)
        else:
            test_data = load_dataset(test_data_path)

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         model = set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_data:
        val_set_size = min(val_set_size, len(val_data["train"])) if val_set_size else len(val_data["train"])
        val_data = val_data["train"].select(range(val_set_size)).map(generate_and_tokenize_prompt)
        if train_data:
            if "num_train_samples" in kwargs and kwargs["num_train_samples"] < len(train_data["train"]):
                train_data = train_data["train"].shuffle().select(range(kwargs["num_train_samples"]))
                if kwargs.get("train_data_path_2", None):
                    train_data_2 = load_dataset("json", data_files=kwargs.get("train_data_path_2"))["train"]
                    train_data = concatenate_datasets([train_data, train_data_2])
                    print(
                        f"Loaded {len(train_data_2)} samples from {kwargs.get('train_data_path_2')} and concatenated to {train_data_path} making {len(train_data)} total samples."
                    )
                train_data = train_data.map(generate_and_tokenize_prompt)

            else:
                train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
    elif val_set_size and val_set_size > 0 and train_data:  # no val data, but val set size is set
        train_val = train_data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=random_seed)
        val_data = train_val["test"].map(generate_and_tokenize_prompt)
        if "num_train_samples" in kwargs and kwargs["num_train_samples"] < len(train_data["train"]):
            train_data = (
                train_data["train"]
                .shuffle()
                .select(range(kwargs["num_train_samples"]))
                .map(generate_and_tokenize_prompt)
            )
        else:
            train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
    elif train_data:
        if "num_train_samples" in kwargs and kwargs["num_train_samples"] < len(train_data["train"]):
            train_data = (
                train_data["train"]
                .shuffle()
                .select(range(kwargs["num_train_samples"]))
                .map(generate_and_tokenize_prompt)
            )
        else:
            train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if test_data:
        if test_set_size:
            test_data = DataLoader(
                test_data["train"].select(range(test_set_size)), batch_size=micro_batch_size, shuffle=False
            )
        else:
            test_data = DataLoader(test_data["train"], batch_size=micro_batch_size, shuffle=False)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    val_set_size = 0 if val_set_size is None else val_set_size
    from transformers import EarlyStoppingCallback, IntervalStrategy

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        # peft_config=peft_config,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=min(100, max(4, int(len(train_data) // batch_size))),
            num_train_epochs=num_epochs,
            max_steps=kwargs.get("max_steps", -1),
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=10,
            optim="adamw_torch",  # "adamw_bnb_8bit" if bf16 else "adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=min(100, max(20, int(len(train_data) // batch_size))) if val_set_size > 0 else None,
            save_steps=min(100, max(20, int(len(train_data) // batch_size))) if val_set_size > 0 else None,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            metric_for_best_model="eval_loss" if val_set_size > 0 else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            hub_private_repo=True,
            hub_model_id=hub_model_id,
            push_to_hub=True if hub_model_id is not None else False,
            dataloader_num_workers=8,
            # Specific args
            # predict_with_generate=False,  # seems buggy
            # generation_num_beams=1,
            # generation_max_length=256,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    from peft.tuners.lora import LoraLayer

    # for name, module in trainer.model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if bf16:
    #             module = module.to(torch.bfloat16)
    #         else:
    #             module = module.to(torch.float16)
    #     if "norm" in name:
    #         module = module.to(torch.float32)
    #
    #     if "lm_head" in name or "embed_tokens" in name:
    #         if hasattr(module, "weight"):
    #             if bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    #             elif module.weight.dtype == torch.float32:
    #                 module = module.to(torch.float16)

    for name, module in trainer.model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # Should not be useful anymore, but just in case
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # Verifying the datatypes.
    dtypes = {}
    for _, p in trainer.model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)
    #

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if train_data:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()

        model = model.merge_and_unload()
        # trainer.save_state()

        model.config.use_cache = True

        if save_whole_model:
            model.save_pretrained(output_dir)

        tokenizer.save_pretrained(output_dir)

        if push_to_hub and hub_model_id is not None:
            tokenizer.push_to_hub(hub_model_id)
            model.push_to_hub(hub_model_id)

        print("\n If there's a warning about missing keys above, please disregard :)")

    if test_data:
        model.eval()
        model.config.use_cache = True
        generation_params = {
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.75),
            "length_pen": kwargs.get("length_pen", 1.0),
            "num_beams": kwargs.get("num_beams", 1),
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "top_k": kwargs.get("top_k", 40),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "device": model.device,
            "batch_size": max(1, predict_batch_size),
            "max_input_length": kwargs.get("max_input_length", 512),
            # Half the batch size for generation since we store all the generated tokens and kv cache
        }

        output_predict_file = os.path.join(output_dir, pred_file_name)
        if trainer.is_world_process_zero():
            print("***** Predicting *****")
            with torch.autocast("cuda"):
                for batch in tqdm(test_data, desc="Generating predictions"):
                    preds = evaluate(batch["instruction"], batch["input"], model, tokenizer, **generation_params)
                    with open(output_predict_file, "a+") as f:
                        for ins, inp, ans, pred in zip(batch["instruction"], batch["input"], batch["output"], preds):
                            f.write(
                                json.dumps({"instruction": ins, "input": inp, "output": ans, "prediction": pred}) + "\n"
                            )

            # trainer.log   _metrics("predict", metrics)
            # trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    fire.Fire(train)
