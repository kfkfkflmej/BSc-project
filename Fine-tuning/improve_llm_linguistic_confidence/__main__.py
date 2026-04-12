import os
from omegaconf import OmegaConf
import logging
import hydra

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    ###################
    # LoRA
    ###################
    lora_config = LoraConfig(
        r=cfg.mapper.lora_r,
        lora_alpha=cfg.mapper.lora_alpha,
        lora_dropout=cfg.mapper.lora_dropout,
        task_type=cfg.mapper.lora_task_type,
        target_modules=cfg.mapper.lora_target_modules
    )

    ###################
    # tokenizer
    ###################
    tokenizer = AutoTokenizer.from_pretrained(
    cfg.mapper.model_base_model,
    use_fast=True,
    token=os.environ['HF_TOKEN']
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    ###################
    # dataset
    ###################
    dataset = load_dataset(cfg.mapper.dataset_filetype, data_files=cfg.mapper.dataset_filepath)   
    # def formatting_prompts_func(example):
    #      return {
    #     "prompt": [{"role": "user", "content": example["problem"]}],
    #     "completion": [{"role": "assistant", "content": example["sentence"]}],
    # }
    #dataset = dataset.map(formatting_prompts_func, remove_columns=["problem", "sentence"])

    def formatting_func(example):
        return {
            "text": f"User: {example['problem']}\nAssistant: {example['sentence']}"
        }

    dataset = dataset.map(formatting_func, remove_columns=["problem", "sentence"])
    

    #dataset = dataset.map(formatting_prompts_func, remove_columns=["problem", "sentence"])

    for split, ds in dataset.items():
        ds.to_json(f"output_{split}.jsonl")

    logging.info(dataset)

    ###################
    # Base_model
    ###################
    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg.mapper.model_base_model, 
    #     load_in_8bit=cfg.mapper.model_load_in_8bit,
    #     device_map=cfg.mapper.model_device_map,
    # )

    

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.mapper.model_base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    ###################
    # SFT args
    ###################
    sft_args = SFTConfig(
        run_name=cfg.mapper.name,
        output_dir=cfg.mapper.sft_output_dir,
        per_device_train_batch_size=cfg.mapper.sft_per_device_train_batch_size,
        gradient_accumulation_steps=cfg.mapper.sft_gradient_accumulation_steps,
        learning_rate=cfg.mapper.sft_learning_rate,
        num_train_epochs=cfg.mapper.sft_num_train_epochs,
        seed=cfg.mapper.sft_seed,
        fp16=cfg.mapper.sft_fp16,
        bf16=cfg.mapper.sft_bf16,
        max_length=cfg.mapper.sft_max_length,
        packing=cfg.mapper.sft_packing,
        warmup_steps=cfg.mapper.sft_warmup_steps,
        gradient_checkpointing=cfg.mapper.sft_gradient_checkpointing,
    )

    ###################
    # Train
    ###################
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,       # needed to tokenize the "text" field
        args=sft_args,
        train_dataset=dataset['train'],
        peft_config=lora_config,
        #formatting_func=formatting_prompts_func,
    )

    print_trainable_parameters(trainer.model)

    trainer.train()

    logging.info("Saving last checkpoint of the model")
    trainer.save_model(sft_args.output_dir)


if __name__ == "__main__":
    main()