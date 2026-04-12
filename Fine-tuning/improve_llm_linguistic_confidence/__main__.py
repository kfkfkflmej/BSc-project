import os
import logging

from omegaconf import OmegaConf
import hydra
import torch
from datasets import Image, load_dataset
from PIL import Image as PILImage
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

@hydra.main(config_path="configs", config_name="config", version_base=None)

def main(cfg):
    dataset = load_dataset(cfg.mapper.dataset_filetype, data_files=cfg.mapper.dataset_filepath)
    model_id = "cfg.mapper.model_base_model"

    def formatting_prompts_func(example, processor=None):
        text = processor.apply_chat_template(...)
        return {"text": text}

    processor = AutoProcessor.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    
    dataset = dataset.map(
        formatting_prompts_func,
        fn_kwargs={"processor": processor},
        remove_columns=["problem", "sentence"]
    )



    # Convert dataset to OAI messages
    dataset = dataset.map(formatting_prompts_func, remove_columns=["problem", "sentence"])
    logging.info(dataset)

    
    # Hugging Face model id
    model_id = "cfg.mapper.model_base_model" # @param ["google/gemma-4-E2B","google/gemma-4-E4B"] {"allow-input":true}


    # Define model init arguments
    model_kwargs = dict(
        dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
        device_map="auto", # Let torch decide how to load the model
    )

    # BitsAndBytesConfig int-4 config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["dtype"],
        bnb_4bit_quant_storage=model_kwargs["dtype"],
    )

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
     # Load the Instruction Tokenizer to use the official Gemma template

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
        ensure_weight_tying=True,
    )

    args = SFTConfig(

        run_name=cfg.mapper.name,
        output_dir=cfg.mapper.sft_output_dir,     # directory to save and repository id
        num_train_epochs=cfg.mapper.sft_num_train_epochs,                    # number of training epochs
        per_device_train_batch_size=cfg.mapper.sft_per_device_train_batch_size,            # batch size per device during training
        gradient_accumulation_steps=cfg.mapper.sft_gradient_accumulation_steps,
        learning_rate=cfg.mapper.sft_learning_rate,
        bf16=True,                                  # use bfloat16 precision
        dataset_text_field="",                      # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
        remove_unused_columns = False,             # important for collator
        max_length=cfg.mapper.sft_max_length,
        packing=cfg.mapper.sft_packing,
        warmup_steps=cfg.mapper.sft_warmup_steps,
        gradient_checkpointing=cfg.mapper.sft_gradient_checkpointing,
        fp16=cfg.mapper.sft_fp16,
    )

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask tokens for not being used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == processor.tokenizer.boi_token_id] = -100
        labels[labels == processor.tokenizer.image_token_id] = -100
        labels[labels == processor.tokenizer.eoi_token_id] = -100

        batch["labels"] = labels
        return batch

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # Start training, the model will be automatically saved to the Hub and the output directory
    trainer.train()

    # Save the final model again to the Hugging Face Hub
    trainer.save_model()

if __name__ == "__main__":
    main()