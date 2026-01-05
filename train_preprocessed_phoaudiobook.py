"""
Train Chatterbox TTS on preprocessed PhoAudiobook data.
This script is optimized for A100 GPU with all acceleration techniques.

Use this script AFTER running preprocess_for_training_phoaudiobook.py
"""

import os
import json
import torch
import time
import glob
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback

# Internal Modules
from src.config_phoaudiobook import PhoAudiobookConfig
from src.dataset_parquet import ParquetDataset
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.utils import setup_logger, check_pretrained_models

# Chatterbox Imports
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxPhoAudiobookFinetune-Preprocessed")


class PreprocessedDataset(Dataset):
    """Dataset that loads preprocessed .pt files."""
    
    def __init__(self, preprocessed_dir, max_samples=None):
        self.preprocessed_dir = preprocessed_dir
        
        # Get all .pt files
        all_files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pt")))
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(all_files):
            all_files = all_files[:max_samples]
        
        self.files = all_files
        logger.info(f"Found {len(self.files)} preprocessed samples")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """Load a preprocessed sample."""
        file_path = self.files[idx]
        
        try:
            # Load preprocessed data
            data = torch.load(file_path, map_location='cpu')
            
            # Convert tensors to correct format
            speech_tokens = data['speech_tokens'].long()
            speaker_emb = data['speaker_emb'].float()
            prompt_tokens = data['prompt_tokens'].long()
            text_tokens = data['text_tokens'].long()
            
            return {
                'speech_tokens': speech_tokens,
                'speaker_emb': speaker_emb,
                'prompt_tokens': prompt_tokens,
                'text_tokens': text_tokens,
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise


def data_collator_preprocessed(batch):
    """Collate preprocessed samples into batches."""
    speech_tokens = torch.stack([item['speech_tokens'] for item in batch])
    speaker_emb = torch.stack([item['speaker_emb'] for item in batch])
    prompt_tokens = torch.stack([item['prompt_tokens'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    
    return {
        'speech_tokens': speech_tokens,
        'speaker_emb': speaker_emb,
        'prompt_tokens': prompt_tokens,
        'text_tokens': text_tokens,
    }


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


class EpochEndCallback(TrainerCallback):
    """Callback to save additional checkpoints at end of each epoch."""
    
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


def main():
    
    cfg = PhoAudiobookConfig()
    
    logger.info("=" * 60)
    logger.info("STARTING CHATTERBOX TRAINING ON PREPROCESSED DATA")
    logger.info("=" * 60)
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")
    logger.info(f"Dataset: PhoAudiobook (Vietnamese) - PREPROCESSED")
    
    # Log GPU information
    log_gpu_info()
    
    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        logger.error("Pretrained models not found. Please run setup.py first.")
        import sys
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
    
    # 5. LOAD PREPROCESSED DATASET
    logger.info("=" * 60)
    logger.info("LOADING PREPROCESSED DATASET")
    logger.info("=" * 60)
    
    if not os.path.exists(cfg.preprocessed_dir):
        logger.error(f"Preprocessed directory not found: {cfg.preprocessed_dir}")
        logger.error("")
        logger.error("Please run: python preprocess_for_training_phoaudiobook.py")
        logger.error("")
        logger.error("Before training: python train_preprocessed_phoaudiobook.py")
        import sys
        sys.exit(1)
    
    # Count preprocessed files
    preprocessed_files = glob.glob(os.path.join(cfg.preprocessed_dir, "*.pt"))
    num_preprocessed = len(preprocessed_files)
    
    if cfg.max_samples is not None and cfg.max_samples < num_preprocessed:
        num_preprocessed = cfg.max_samples
    
    logger.info(f"Preprocessed directory: {cfg.preprocessed_dir}")
    logger.info(f"Number of preprocessed samples: {num_preprocessed}")
    
    # Check if using subset
    if cfg.max_samples is not None:
        logger.info(f"Using all preprocessed samples")
    else:
        logger.info(f"Using subset: {cfg.max_samples} samples (from {num_preprocessed} total)")
    
    # Load preprocessed dataset
    train_ds = PreprocessedDataset(cfg.preprocessed_dir, max_samples=cfg.max_samples)
    
    steps_per_epoch = num_preprocessed // (cfg.batch_size * cfg.grad_accum)
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
    
    logger.info(f"Training samples: {num_preprocessed}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    
    if hasattr(cfg, 'save_steps_fixed') and cfg.save_steps_fixed is not None:
        logger.info(f"Saving checkpoint every {save_steps} steps")
        logger.info(f"Estimated time to first checkpoint: ~{save_steps * 0.05 / 3600:.1f} hours")
    else:
        logger.info(f"Saving checkpoint every {getattr(cfg, 'save_freq_epochs', 10)} epochs ({save_steps} steps)")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    model_wrapper = ChatterboxTrainerWrapper(new_t3_model, config=cfg)
    
    # Enable torch.compile for A100 acceleration (PyTorch 2.0+)
    use_torch_compile = getattr(cfg, 'use_torch_compile', True)
    if use_torch_compile and torch.cuda.is_available():
        try:
            torch_version = torch.__version__
            major_version = int(torch_version.split('.')[0])
            
            if major_version >= 2:
                logger.info("=== Enabling torch.compile for acceleration ===")
                logger.info(f"PyTorch {torch_version} detected - compiling model...")
                
                # Compile model wrapper for faster execution
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
    
    # Create callbacks
    callbacks = [
        SpeedMonitorCallback(),  # Monitor training speed
        MemoryMonitorCallback(log_interval=50),  # Monitor GPU memory
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
    
    # TRAINING ARGUMENTS - Optimized for A100 GPU
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
        max_steps=total_training_steps,  # Required for DataLoader
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
    )
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator_preprocessed
    )
    
    logger.info("Starting Training Loop...")
    logger.info(f"Effective batch size: {cfg.batch_size * cfg.grad_accum}")
    
    # Create trainer
    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_dataloader,
        callbacks=callbacks
    )
    
    # Train
    trainer.train(resume_from_checkpoint=resume_path)
    
    # 9. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    filename = "t3_turbo_phoaudiobook.pt" if cfg.is_turbo else "t3_phoaudiobook.pt"
    final_model_path = os.path.join(cfg.output_dir, filename)

    torch.save(new_t3_model.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("Fine-tuning complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

