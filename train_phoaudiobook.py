import os
import sys
import glob
import torch
import json
import time
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


def log_gpu_info():
    """Log GPU information and capabilities."""
    if not torch.cuda.is_available():
        logger.info("No CUDA GPU available")
        return False
    
    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    
    logger.info(f"=== GPU Information ===")
    logger.info(f"GPU: {device_properties.name}")
    logger.info(f"Total Memory: {device_properties.total_memory / 1e9:.2f} GB")
    logger.info(f"Compute Capability: {device_properties.major}.{device_properties.minor}")
    logger.info(f"Multi-processors: {device_properties.multi_processor_count}")
    logger.info(f"=====================")
    
    # Check for BF16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    logger.info(f"BF16 Supported: {'YES ✓' if bf16_supported else 'NO ✗'}")
    
    # Check for Flash Attention support
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        logger.info(f"Flash Attention 2: Available ✓")
    except ImportError:
        logger.info(f"Flash Attention 2: Not Available (install with: pip install flash-attn)")
    
    # Check PyTorch version for compile support
    torch_version = torch.__version__
    compile_supported = int(torch_version.split('.')[0]) >= 2
    logger.info(f"PyTorch {torch_version}")
    logger.info(f"Torch Compile: {'Supported ✓' if compile_supported else 'Not Supported ✗'}")
    
    return True


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


class SpeedMonitorCallback(TrainerCallback):
    """Callback to monitor and log training speed statistics."""
    
    def __init__(self):
        self.last_log_time = None
        self.last_log_step = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after each logging step."""
        current_time = time.time()
        current_step = state.global_step
        
        if self.last_log_time is not None and self.last_log_step is not None:
            steps_since_last_log = current_step - self.last_log_step
            time_since_last_log = current_time - self.last_log_time
            
            if steps_since_last_log > 0 and time_since_last_log > 0:
                speed = steps_since_last_log / time_since_last_log  # steps per second
                time_per_step = time_since_last_log / steps_since_last_log  # seconds per step
                
                # Log speed metrics
                logger.info(f"Training Speed: {speed:.2f} steps/sec ({time_per_step:.2f} sec/step)")
                
                # Estimate remaining time
                if state.max_steps > 0:
                    steps_remaining = state.max_steps - current_step
                    time_remaining_hours = (steps_remaining / speed) / 3600
                    logger.info(f"Estimated time remaining: {time_remaining_hours:.1f} hours")
        
        self.last_log_time = current_time
        self.last_log_step = current_step


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor GPU memory usage during training."""
    
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.last_log_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        if state.global_step - self.last_log_step >= self.log_interval:
            if torch.cuda.is_available():
                # Get memory usage
                allocated = torch.cuda.memory_allocated() / 1e9  # GB
                reserved = torch.cuda.memory_reserved() / 1e9  # GB
                max_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
                
                logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {max_allocated:.2f} GB max")
                
                # Reset max memory periodically
                if state.global_step % (self.log_interval * 10) == 0:
                    torch.cuda.reset_peak_memory_stats()
            
            self.last_log_step = state.global_step


class SaveInitialModelCallback(TrainerCallback):
    """Callback to save initial model state before training starts."""
    
    def __init__(self, model, output_dir): 
        self.model = model
        self.output_dir = output_dir
        self.initial_model_saved = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called once at the beginning of training."""
        if not self.initial_model_saved:
            logger.info("=" * 60)
            logger.info("SAVING INITIAL MODEL CHECKPOINT (STEP 0)")
            logger.info("=" * 60)
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.output_dir, "checkpoint-0")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model weights manually
            model_state = self.model.state_dict()
            save_path = os.path.join(checkpoint_dir, "model.safetensors")
            save_file(model_state, save_path)
            
            # Save training arguments
            args_dict = {
                "output_dir": self.output_dir,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "save_steps": args.save_steps,
                "save_strategy": args.save_strategy,
                "logging_steps": args.logging_steps,
                "logging_strategy": args.logging_strategy,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "gradient_checkpointing": args.gradient_checkpointing,
            }
            
            args_path = os.path.join(checkpoint_dir, "training_args.bin")
            with open(args_path, "w") as f:
                json.dump(args_dict, f)
            
            # Save trainer state
            trainer_state = {
                "epoch": 0.0,
                "global_step": 0,
                "log_history": [],
            }
            
            state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            with open(state_path, "w") as f:
                json.dump(trainer_state, f)
            
            # Save config (only if model has config attribute)
            if hasattr(self.model, 'config') and self.model.config is not None:
                try:
                    config = self.model.config
                    config_path = os.path.join(checkpoint_dir, "config.json")
                    if hasattr(config, 'to_dict'):
                        config_dict = config.to_dict()
                    else:
                        config_dict = str(config)
                    with open(config_path, "w") as f:
                        json.dump(config_dict, f)
                    logger.info(f"  - Model config: {config_path}")
                except Exception as e:
                    logger.warning(f"Could not save model config: {e}")
            
            self.initial_model_saved = True
            
            logger.info(f"✓ Initial checkpoint saved to: {checkpoint_dir}")
            logger.info(f"  - Model weights: {save_path}")
            logger.info(f"  - Training args: {args_path}")
            logger.info(f"  - Trainer state: {state_path}")
            logger.info("=" * 60)


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


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory.
    
    Returns:
        str or None: Path to the latest checkpoint directory, or None if no checkpoint exists.
    """
    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # Extract step numbers from checkpoint names
    checkpoint_with_steps = []
    for checkpoint_dir in checkpoint_dirs:
        try:
            # Extract the number from checkpoint-XXXXX
            step_num = int(checkpoint_dir.split("-")[-1])
            checkpoint_with_steps.append((step_num, checkpoint_dir))
        except (ValueError, IndexError):
            # Skip malformed checkpoint names
            continue
    
    if not checkpoint_with_steps:
        return None
    
    # Sort by step number (descending) and return the latest
    checkpoint_with_steps.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_checkpoint = checkpoint_with_steps[0]
    
    logger.info(f"Found latest checkpoint: {latest_checkpoint} (step {latest_step})")
    return latest_checkpoint
  

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
    
    # Log GPU information
    log_gpu_info()
    
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

    # Enable torch.compile for A100 acceleration (PyTorch 2.0+)
    use_torch_compile = getattr(cfg, 'use_torch_compile', True)
    if use_torch_compile and torch.cuda.is_available():
        try:
            torch_version = torch.__version__
            major_version = int(torch_version.split('.')[0])
            
            if major_version >= 2:
                logger.info("=== Enabling torch.compile for acceleration ===")
                logger.info(f"PyTorch {torch_version} detected - compiling model...")
                
                # Compile the model wrapper for faster execution
                # mode="max-autotune" - maximum optimization, slower initial compile but fastest execution
                # mode="reduce-overhead" - balance between compile time and execution speed
                compile_mode = "reduce-overhead"  # Good balance for training
                model_wrapper = torch.compile(
                    model_wrapper,
                    mode=compile_mode,
                    fullgraph=False  # Safe for training
                )
                logger.info(f"✓ Model compiled successfully with mode='{compile_mode}'")
                logger.info("  Expect 20-50% speedup after first few batches")
            else:
                logger.warning(f"PyTorch {torch_version} < 2.0 - torch.compile not available")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
            logger.warning("Continuing without torch.compile")

    # Check for Flash Attention 2
    use_flash_attention = getattr(cfg, 'use_flash_attention', True)
    if use_flash_attention:
        try:
            import flash_attn
            logger.info("✓ Flash Attention 2 is available - will be used if supported by model")
        except ImportError:
            logger.info("Flash Attention 2 not available - install with: pip install flash-attn")

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
    callbacks = [
        SaveInitialModelCallback(model_wrapper, cfg.output_dir),  # Save initial model at step 0
        inference_callback, 
        SpeedMonitorCallback(),  # Monitor training speed
        MemoryMonitorCallback(log_interval=50)  # Monitor GPU memory
    ]
    if save_at_epoch_end:
        epoch_callback = EpochEndCallback(steps_per_epoch=steps_per_epoch)
        callbacks.append(epoch_callback)

    # Check for resume from checkpoint
    resume_path = None
    if getattr(cfg, 'resume_from_checkpoint', False):
        logger.info("Resume from checkpoint enabled. Searching for latest checkpoint...")
        resume_path = find_latest_checkpoint(cfg.output_dir)
        if resume_path:
            logger.info(f"Will resume training from: {resume_path}")
        else:
            logger.warning(f"No checkpoint found in {cfg.output_dir}. Starting fresh training.")

    # 8. TRAINING ARGUMENTS - Optimized for A100 GPU
    # Determine precision settings
    use_bf16 = getattr(cfg, 'use_bf16', True) and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and torch.cuda.is_available()
    
    logger.info(f"=== Precision Settings ===")
    logger.info(f"BF16: {use_bf16}")
    logger.info(f"FP16: {use_fp16}")
    
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
        dataloader_num_workers=4,  # Parallel data loading for better throughput
        report_to=["tensorboard"],
        
        # Precision - A100 optimized
        bf16=use_bf16,  # Native A100 support, better than FP16
        fp16=use_fp16,
        
        # Checkpointing
        save_total_limit=getattr(cfg, 'save_total_limit', 10),  # Keep last N checkpoints
        gradient_checkpointing=True,  # Reduces VRAM usage by ~60%
        
        # Memory optimization
        dataloader_pin_memory=True,
        
        # Learning rate scheduling - better convergence
        warmup_ratio=0.01,  # 1% warmup
        lr_scheduler_type="cosine",  # Smooth cosine decay
        
        # Performance optimizations
        ddp_find_unused_parameters=False,  # Faster DDP
        # dataloader_prefetch_factor=2,  # Prefetch next batches
        # gradient_accumulation_kwargs=None,
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
    
    # Train with or without resuming from checkpoint
    trainer.train(resume_from_checkpoint=resume_path)

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

