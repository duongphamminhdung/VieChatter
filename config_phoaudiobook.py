from dataclasses import dataclass

@dataclass
class PhoAudiobookConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the pretrained models (in Drive)
    model_dir: str = "/content/drive/MyDrive/VieNP/pretrained_models"
    
    # Path to PhoAudiobook parquet file(s) - USE LOCAL PATH for faster access
    # Using /content/ for Colab (faster than Drive)
    # Files are in /content/phoaudiobook/data/ subdirectory
    parquet_path: str = "/content/phoaudiobook/data"

    # Output directory for finetuned model - Save to Google Drive
    output_dir: str = "/content/drive/MyDrive/VieNP/models"

    # Dataset settings
    dataset_format: str = "phoaudiobook"  # Use PhoAudiobook parquet format
    
                           # Model type
    is_turbo: bool = True  # True for Turbo, False for Normal
    
    # --- Vocabulary ---
    # For Vietnamese, use the same tokenizer as the base model
    # Turbo mode typically uses a larger vocabulary
    new_vocab_size: int = 52260  # Fixed for Vietnamese dataset
    
    # --- Hyperparameters ---
    batch_size: int = 2  # Reduced for Colab free tier
    grad_accum: int = 4  # Effective batch size = 2 * 4 = 8
    learning_rate: float = 5e-5
    num_epochs: int = 30  # Full training run for 30 epochs
    save_steps_fixed: int = 5000  # Save checkpoint every N steps
    save_at_epoch_end: bool = True  # Also save checkpoint at the end of each epoch
    save_total_limit: int = 20  # Keep last N checkpoints (inference files not affected)

    # Inference message (Vietnamese greeting)
    inference_message: str = "Đây là đài Tiếng nói Việt Nam, phát thanh từ Hà Nội. Xin chào tất cả các bạn, tôi là Việt chatter, model text to speech được phát triển bởi Dpmd"
    
    # --- Constraints ---
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 999999  # No truncation (set very high to disable)
    max_speech_len: int = 999999  # No truncation (set very high to disable)
    prompt_duration: float = 3.0
    
                                  # --- PhoAudiobook Specific ---
                                  # Dataset split to process (train, validation, test, or None for all)
    dataset_split: str = "train"  # Options: "train", "validation", "test", None (all files)

    # Limit number of samples for testing (set to None to use all)
    # Start with 1000 samples to test RAM usage
    max_samples: int = None  # e.g., 1000 for quick testing, None for full dataset
    
    # Audio duration filters (in seconds)
    min_audio_duration: float = 1.0
    max_audio_duration: float = None  # No max duration limit
