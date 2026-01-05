import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

# Internal Modules
from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator
from src.dataset_parquet import ParquetDataset, data_collator_parquet, create_sequential_parquet_dataloader
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.utils import setup_logger, check_pretrained_models

# Chatterbox Imports
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def main():
    
    cfg = TrainConfig()
    
    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS
    
    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # We prevent caching during training.
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)


    # --- SPECIAL SETTING FOR TURBO ---
    if cfg.is_turbo:
        logger.info("Turbo Mode: Removing backbone WTE layer...")
        if hasattr(new_t3_model.tfmr, "wte"):
            del new_t3_model.tfmr.wte


    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    # Reload engine components (VoiceEncoder, S3Gen) but inject our new T3
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model 

    # Freeze other components
    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters(): 
        param.requires_grad = False
        
    for param in tts_engine_new.s3gen.parameters(): 
        param.requires_grad = False

    # Enable Training for T3
    tts_engine_new.t3.train()
    for param in tts_engine_new.t3.parameters(): 
        param.requires_grad = True

    if cfg.preprocess and not cfg.use_parquet:

        logger.info("Initializing Preprocess dataset...")

        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)

        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)

    else:
        if cfg.use_parquet:
            logger.info("Using parquet dataset (no preprocessing needed)...")
        else:
            logger.info("Skipping the preprocessing dataset step...")


    # 6. DATASET & WRAPPER
    logger.info("Initializing Dataset...")

    if cfg.use_parquet:
        # Use sequential parquet dataloader for optimal performance
        import multiprocessing as mp
        num_dataloader_workers = min(mp.cpu_count(), 4)

        train_ds, train_dataset = create_sequential_parquet_dataloader(
            cfg,
            tts_engine_new,
            split="train",
            num_samples=cfg.max_samples,
            batch_size=cfg.batch_size,
            num_workers=num_dataloader_workers
        )

        num_train_samples = len(train_dataset)
        logger.info(f"Using parquet dataset with {num_dataloader_workers} workers")
        logger.info(f"Processing partitions sequentially, samples in parallel")
    else:
        # Use traditional preprocessed dataset
        train_ds = ChatterboxDataset(cfg)
        num_train_samples = len(train_ds)
        logger.info(f"Using preprocessed dataset")

    steps_per_epoch = num_train_samples // (cfg.batch_size * cfg.grad_accum)
    save_steps = steps_per_epoch * 10  # Save every 10 epochs

    logger.info(f"Training samples: {num_train_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Saving checkpoint every 10 epochs ({save_steps} steps)")

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3, config=cfg)

    # 7. TRAINING ARGUMENTS
    if cfg.use_parquet:
        # For parquet dataset: workers handled by SequentialPartitionDataset
        data_collator_fn = data_collator_parquet
        dataloader_workers = 0
    else:
        # For traditional dataset: use standard settings
        data_collator_fn = data_collator
        dataloader_workers = 4

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="epoch",
        remove_unused_columns=False, # Required for our custom wrapper
        dataloader_num_workers=dataloader_workers,
        report_to=["tensorboard"],
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=2,
        gradient_checkpointing=True, # This setting theoretically reduces VRAM usage by 60%.
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_fn
    )

    logger.info("Starting Training Loop...")
    trainer.train()


    # 8. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    filename = "t3_turbo_finetuned.safetensors" if cfg.is_turbo else "t3_finetuned.safetensors"
    final_model_path = os.path.join(cfg.output_dir, filename)

    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__": 
    main()
