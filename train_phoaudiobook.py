import os
import sys
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from safetensors.torch import save_file

# Internal Modules
from config_phoaudiobook import PhoAudiobookConfig
from src.dataset import ChatterboxDataset, data_collator
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_phoaudiobook import preprocess_dataset_phoaudiobook
from src.utils import setup_logger, check_pretrained_models

# Chatterbox Imports
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxPhoAudiobookFinetune")


class InferenceCallback(TrainerCallback):
    """Callback to save inference files at each checkpoint."""

    def __init__(self, model, tokenizer, message, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.message = message
        self.output_dir = output_dir

    def on_save(self, args, state, control):
        """Called when a checkpoint is saved."""
        # Only save at actual checkpoint saves (not intermediate saves)
        if state.global_step % args.save_steps == 0:
            self.save_inference_file(state.global_step)

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

    # 6. PREPROCESS DATASET
    if cfg.preprocess:
        logger.info("Preprocessing PhoAudiobook dataset...")
        logger.info(f"Reading from: {cfg.parquet_path}")
        logger.info(f"Output to: {cfg.preprocessed_dir}")
        preprocess_dataset_phoaudiobook(cfg, tts_engine_new)
    else:
        logger.info("Skipping preprocessing (using existing preprocessed data)")
    
    # Check if preprocessed files exist before initializing dataset
    if not os.path.exists(cfg.preprocessed_dir):
        raise FileNotFoundError(
            f"Preprocessed directory does not exist: {cfg.preprocessed_dir}\n"
            f"Please set preprocess=True in config and run again."
        )

    # Count files efficiently (streaming, not loading all into RAM)
    preprocessed_dir = cfg.preprocessed_dir
    file_count       = 0
    try:
        with os.scandir(preprocessed_dir) as entries:
            for entry in entries:
                if entry.name.endswith('.pt'):
                    file_count += 1
    except OSError as e:
        raise FileNotFoundError(f"Cannot read directory {preprocessed_dir}: {e}")

    if file_count == 0:
        raise FileNotFoundError(
            f"No .pt files found in: {preprocessed_dir}\n"
            f"Preprocessing may have failed. Please:\n"
            f"1. Check that preprocess=True in config\n"
            f"2. Verify parquet_path is correct: {cfg.parquet_path}\n"
            f"3. Check preprocessing logs above for errors\n"
            f"4. Try running preprocessing separately to debug"
        )
    else:
        logger.info(f"Found {file_count} preprocessed file(s)")

    # Calculate save steps for epoch-based saving
    # save_steps = (total_samples / (batch_size * grad_accum)) * save_freq_epochs
    num_train_samples = len(preprocessed_files)
    steps_per_epoch = num_train_samples // (cfg.batch_size * cfg.grad_accum)
    save_steps = steps_per_epoch * getattr(cfg, 'save_freq_epochs', 10)

    logger.info(f"Training samples: {num_train_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Saving checkpoint every {getattr(cfg, 'save_freq_epochs', 10)} epochs ({save_steps} steps)")
        
    # 7. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg, num_samples=file_count)

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    # Create inference callback to save Vietnamese greeting at each checkpoint
    inference_callback = InferenceCallback(
        model=tts_engine_new.t3,
        tokenizer=tts_engine_new.tokenizer,
        message=cfg.inference_message,
        output_dir=cfg.output_dir
    )

    # 8. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_num_workers=2,  # Reduced for Colab
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
        data_collator=data_collator,
        callbacks=[inference_callback]  # Add inference callback
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

