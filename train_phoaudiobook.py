import os
import sys
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from safetensors.torch import save_file

# Internal Modules
from config_phoaudiobook import PhoAudiobookConfig
from src.dataset_parquet import ParquetDataset, data_collator_parquet, SequentialPartitionDataset
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.utils import setup_logger, check_pretrained_models

# Chatterbox Imports
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxPhoAudiobookFinetune")


class InferenceCallback(TrainerCallback):
    """Callback to save inference files at each checkpoint."""

    def __init__(self, model, tokenizer, message, output_dir, save_steps=None, steps_per_epoch=None):
        self.model = model
        self.tokenizer = tokenizer
        self.message = message
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.steps_per_epoch = steps_per_epoch

    def on_save(self, args, state, control):
        """Called when a checkpoint is saved."""
        # Use provided save_steps or fall back to args.save_steps
        check_step = self.save_steps if self.save_steps else args.save_steps
        # Only save at actual checkpoint saves (not intermediate saves)
        if state.global_step % check_step == 0:
            self.save_inference_file(state.global_step)


class EpochEndCallback(TrainerCallback):
    """Callback to save additional checkpoints at the end of each epoch."""

    def __init__(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.last_epoch_saved = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        current_epoch = int(state.epoch)
        
        # Only save once per epoch (avoid duplicates)
        if current_epoch > self.last_epoch_saved:
            self.last_epoch_saved = current_epoch
            logger.info(f"Saving epoch-end checkpoint for epoch {current_epoch}")
            # Force save by updating control
            control.should_save = True
            control.should_log = True

    def save_inference_file(self, step):
        """Save an inference-ready file with the Vietnamese message."""
        # Create separate directory for inference samples (not affected by save_total_limit)
        inference_dir = os.path.join(self.output_dir, "inference_samples")
        os.makedirs(inference_dir, exist_ok=True)

        # Tokenize the Vietnamese message
        tokens = self.tokenizer(
            self.message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Save inference-ready file
        inference_file = os.path.join(inference_dir, f"inference_step_{step}.pt")
        torch.save({
            "text_tokens": tokens["input_ids"][0],
            "text": self.message,
            "step": step
        }, inference_file)

        logger.info(f"Saved inference file: {inference_file}")
        logger.info(f"Message: {self.message}")


def main():
    
    cfg = PhoAudiobookConfig()
    
    logger.info("--- Starting Chatterbox Finetuning on PhoAudiobook ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")
    logger.info(f"Dataset: PhoAudiobook (Vietnamese)")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        logger.error("Pretrained models not found. Please run setup.py first.")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS
    
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # Disable cache during training
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
    logger.info("Preparing engine for training...")
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

    # 6. INITIALIZING DATASET (NO PREPROCESSING NEEDED)
    logger.info("Using Sequential ParquetDataset - reads parquet files sequentially and processes audio on-the-fly")
    logger.info(f"Reading from: {cfg.parquet_path}")
    logger.info("No preprocessing step required (saves time and storage)")

    # Validate parquet path exists
    if not os.path.exists(cfg.parquet_path):
        raise FileNotFoundError(
            f"Parquet directory does not exist: {cfg.parquet_path}\n"
            f"Please verify the path in your config file."
        )

    # Initialize sequential dataset (processes partitions one at a time)
    logger.info("Initializing Sequential Dataset...")
    import multiprocessing as mp
    num_dataloader_workers = min(mp.cpu_count(), 4)

    from src.dataset_parquet import SequentialPartitionDataset, ParquetDataset

    # Create base ParquetDataset
    parquet_dataset = ParquetDataset(
        cfg,
        tts_engine=tts_engine_new,
        split="train",
        num_samples=cfg.max_samples
    )

    # Create sequential iterable dataset
    train_ds = SequentialPartitionDataset(parquet_dataset)

    num_train_samples    = len(train_ds)
    steps_per_epoch      = num_train_samples // (cfg.batch_size * cfg.grad_accum)
    total_training_steps = steps_per_epoch * cfg.num_epochs
    
    # Use fixed step interval if configured, otherwise use epoch-based frequency
    if hasattr(cfg, 'save_steps_fixed') and cfg.save_steps_fixed is not None:
        save_steps = cfg.save_steps_fixed
        logger.info(f"Using fixed step interval: {save_steps} steps per checkpoint")
    else:
        save_steps = steps_per_epoch * getattr(cfg, 'save_freq_epochs', 10)
        logger.info(f"Using epoch-based frequency: {getattr(cfg, 'save_freq_epochs', 10)} epochs ({save_steps} steps)")
    
    # Check if saving at epoch end is enabled
    save_at_epoch_end = getattr(cfg, 'save_at_epoch_end', False)
    if save_at_epoch_end:
        logger.info("Additional checkpoints will be saved at the end of each epoch")

    logger.info(f"Training samples: {num_train_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    
    if hasattr(cfg, 'save_steps_fixed') and cfg.save_steps_fixed is not None:
        logger.info(f"Saving checkpoint every {save_steps} steps")
        logger.info(f"Estimated time to first checkpoint: ~{save_steps * 1.78 / 3600:.1f} hours")
    else:
        logger.info(f"Saving checkpoint every {getattr(cfg, 'save_freq_epochs', 10)} epochs ({save_steps} steps)")
    
    logger.info(f"Using {num_dataloader_workers} workers for parallel processing within each partition")

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3, config=cfg)

    # Create inference callback to save Vietnamese greeting at each checkpoint
    # Pass save_steps for consistent checkpoint frequency
    inference_callback = InferenceCallback(
        model=tts_engine_new.t3,
        tokenizer=tts_engine_new.tokenizer,
        message=cfg.inference_message,
        output_dir=cfg.output_dir,
        save_steps=save_steps,
        steps_per_epoch=steps_per_epoch
    )

    # Create epoch-end callback if enabled
    callbacks = [inference_callback]
    if save_at_epoch_end:
        epoch_callback = EpochEndCallback(steps_per_epoch=steps_per_epoch)
        callbacks.append(epoch_callback)

    # 8. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=total_training_steps,  # Required for IterableDataset
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Already handled by SequentialPartitionDataset
        report_to=["tensorboard"],
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=getattr(cfg, 'save_total_limit', 10),  # Keep last N checkpoints
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_parquet,
        callbacks=callbacks
    )

    logger.info("Starting Training Loop...")
    logger.info(f"Effective batch size: {cfg.batch_size * cfg.grad_accum}")
    trainer.train()

    # 9. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    filename = "t3_turbo_phoaudiobook.safetensors" if cfg.is_turbo else "t3_phoaudiobook.safetensors"
    final_model_path = os.path.join(cfg.output_dir, filename)

    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("Fine-tuning complete!")


if __name__ == "__main__": 
    main()

