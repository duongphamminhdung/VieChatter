import os
import torch
import torchaudio
import dask.dataframe as dd
import io
from tqdm import tqdm
from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger


logger = setup_logger(__name__)


def preprocess_dataset_phoaudiobook(config, tts_engine: ChatterboxTTS): 
    """
    Preprocess PhoAudiobook dataset from parquet files.
    
    Expected parquet columns:
    - audio: Audio data or path to audio files
    - text: Transcription text
    - speaker_id: Speaker identifier (optional)
    - split: Dataset split (train/val/test, optional)
    
    Features:
    - Processes data in chunks to avoid RAM overflow
    - Skips already processed files (can resume if interrupted)
    - Supports max_samples limit for testing
    
    Args:
        config: Training configuration object
        tts_engine: ChatterboxTTS engine instance
    """
    
    # Read parquet file
    logger.info(f"Reading parquet file from: {config.parquet_path}")

    # Check if parquet path exists
    if os.path.isfile(config.parquet_path):
        logger.info(f"Parquet file found: {config.parquet_path}")
        parquet_files = [config.parquet_path]
    elif os.path.isdir(config.parquet_path):
        logger.info(f"Parquet directory found: {config.parquet_path}")

        # Define split patterns
        splits = {
            'train': 'train-*.parquet',
            'validation': 'validation-*.parquet',
            'test': 'test-*.parquet'
        }

        if config.dataset_split and config.dataset_split in splits:
            # Read specific split
            pattern = splits[config.dataset_split]
            import glob
            parquet_files = sorted(glob.glob(os.path.join(config.parquet_path, pattern)))
            logger.info(f"Using split '{config.dataset_split}' with pattern: {pattern}")
        else:
            # List all parquet files
            import glob
            parquet_files = sorted(glob.glob(os.path.join(config.parquet_path, '*.parquet')))
            logger.info(f"Reading all parquet files in directory")

        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in directory: {config.parquet_path}")
        logger.info(f"Found {len(parquet_files)} parquet file(s): {[os.path.basename(f) for f in parquet_files]}")
    else:
        raise FileNotFoundError(f"Parquet path not found: {config.parquet_path}")

    # Create output directory
    os.makedirs(config.preprocessed_dir, exist_ok=True)
    logger.info(f"Output directory: {config.preprocessed_dir}")

    # Read parquet file(s) with dask
    try:
        df = dd.read_parquet(parquet_files)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file(s): {e}")
    
    # Get the total number of rows without loading into memory
    # logger.info("Calculating dataset size...")
    # try:
    #     total_rows = len(df)
    #     logger.info(f"Total rows in parquet file(s): {total_rows}")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to calculate dataset size: {e}")
    
    # Apply max_samples limit if specified
    # if config.max_samples is not None:
    #     logger.info(f"Limiting to {config.max_samples} samples (from {total_rows} total)")
    #     total_rows = min(total_rows, config.max_samples)
    
      # logger.info(f"Processing {total_rows} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    tts_engine.ve.eval()
    tts_engine.s3gen.eval()

    success_count = 0
    skipped_count = 0
    already_processed_count = 0
    file_counter = 0  # Counter for sequential filename numbering

    # Process in chunks to avoid RAM overflow
    # Iterate through partitions without loading everything into memory
    logger.info("Processing data in chunks to avoid RAM overflow...")
    
    # Get number of partitions
    n_partitions = df.npartitions
    logger.info(f"Processing {n_partitions} partition(s)...")
    
    # Process each partition
    samples_processed = 0
    for partition_idx in range(n_partitions):
        if config.max_samples is not None and samples_processed >= config.max_samples:
            logger.info(f"Reached max_samples limit ({config.max_samples}), stopping.")
            break
            
        logger.info(f"Processing partition {partition_idx + 1}/{n_partitions}...")
        
        # Get only this partition as a pandas dataframe
        partition_df = df.get_partition(partition_idx).compute()
        
        # Process each row in this partition
        for local_idx, (parquet_idx, row) in enumerate(partition_df.iterrows()):
            # Check if we've reached max_samples limit
            if config.max_samples is not None and samples_processed >= config.max_samples:
                logger.info(f"Reached max_samples limit ({config.max_samples}), stopping.")
                break
            
            try:
                # Get audio and text from parquet row
                audio_data = row.get('audio')
                text = row.get('text', '')
                
                if not text or str(text).strip() == '':
                    samples_processed += 1
                    skipped_count += 1
                    continue  # Skip early, don't increment file_counter
                
                # Handle audio - can be path or embedded data
                if isinstance(audio_data, str) and os.path.exists(audio_data):
                    # Audio is a file path
                    wav, sr = torchaudio.load(audio_data)
                elif hasattr(audio_data, 'array'):
                    # Audio is embedded (like from HuggingFace datasets)
                    import numpy as np
                    audio_array = np.array(audio_data.array)
                    wav = torch.from_numpy(audio_array).unsqueeze(0).float()
                    sr = audio_data.sampling_rate if hasattr(audio_data, 'sampling_rate') else 24000
                elif isinstance(audio_data, dict):
                    import numpy as np
                    if 'array' in audio_data:
                        # Audio is a dict with array and sampling_rate
                        audio_array = np.array(audio_data['array'])
                        wav = torch.from_numpy(audio_array).unsqueeze(0).float()
                        sr = audio_data.get('sampling_rate', 24000)
                    elif 'bytes' in audio_data:
                        # Audio is raw WAV bytes in dict
                        wav_bytes = audio_data['bytes']
                        wav_io = io.BytesIO(wav_bytes)
                        wav, sr = torchaudio.load(wav_io)
                    else:
                        # Log the type and structure of audio_data for debugging
                        logger.warning(
                            f"Could not parse audio data at row {global_idx}. "
                            f"Audio is dict but no 'array' or 'bytes' key. Keys: {list(audio_data.keys())}, "
                            f"skipping..."
                        )
                        skipped_count += 1
                        continue
                else:
                    # Log the type and structure of audio_data for debugging
                    logger.warning(
                        f"Could not parse audio data at row {file_counter}. "
                        f"Audio data type: {type(audio_data)}, "
                        f"str representation: {str(audio_data)[:200] if audio_data else 'None'}, "
                        f"skipping..."
                    )
                    skipped_count += 1
                    samples_processed += 1
                    continue  # Skip early, don't increment file_counter
                
                # Convert to mono if stereo
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sr != S3_SR:
                    resampler = torchaudio.transforms.Resample(sr, S3_SR)
                    wav = resampler(wav)
                
                wav = wav.to(device)
                
                # Skip if audio is too short
                if wav.shape[1] < 0.5 * S3_SR:  # Less than 0.5 seconds
                    skipped_count += 1
                    samples_processed += 1
                    continue  # Skip early, don't increment file_counter
                
                with torch.no_grad():
                    # Extract speaker embedding
                    wav_np = wav.cpu().squeeze().numpy()
                    spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                    speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()

                    # Tokenize full speech
                    s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                    speech_tokens = s_tokens.squeeze().cpu()

                    # Extract prompt (first N seconds)
                    prompt_samples = int(config.prompt_duration * S3_SR)
                    
                    if wav.shape[1] < prompt_samples:
                        prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                    else:
                        prompt_wav = wav[:, :prompt_samples]
                    
                    p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                    prompt_tokens = p_tokens.squeeze().cpu()

                # Clean and normalize text
                raw_text = str(text).strip()
                clean_text = punc_norm(raw_text)
                
                # Tokenize text
                if config.is_turbo:
                    token_output = tts_engine.tokenizer(clean_text, return_tensors="pt")
                    text_tokens = token_output.input_ids[0]
                else:
                    text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()
                
                # Create unique filename
                speaker_id = row.get('speaker_id', f'unk_{file_counter}')
                filename = f"speaker_{speaker_id}_{file_counter:06d}"
                save_path = os.path.join(config.preprocessed_dir, f"{filename}.pt")
                
                    # Skip if file already exists
                if os.path.exists(save_path):
                    already_processed_count += 1
                    samples_processed += 1
                    file_counter += 1  # Increment for existing files too
                    # if already_processed_count % 100 == 0:
                    #     print(f"Progress: {success_count} new, {already_processed_count} already exist | Total: {samples_processed}/{total_rows}")
                    continue
                
                # Save preprocessed data
                torch.save({
                    "speech_tokens": speech_tokens,
                    "speaker_emb": speaker_emb,
                    "prompt_tokens": prompt_tokens,
                    "text_tokens": text_tokens,
                }, save_path)

                success_count += 1
                samples_processed += 1
                file_counter += 1  # Increment for successfully processed files

                  # Update progress
                if success_count % 100 == 0:
                    print(f"Progress: {success_count} new, {already_processed_count} already exist | Skipped: {skipped_count}")

            except Exception as e:
                logger.error(f"Error processing row {file_counter}: {e}")
                skipped_count += 1
                samples_processed += 1
                continue  # Skip early, don't increment file_counter
        
        # Clear memory after processing each partition
        del partition_df
        if partition_idx < n_partitions - 1:
            import gc
            gc.collect()
    
    logger.info(f"Preprocessing completed!")
    logger.info(f"Successfully processed: {success_count} new samples")
    logger.info(f"Already existed: {already_processed_count} samples (skipped)")
    logger.info(f"Skipped (errors): {skipped_count} samples")
    # logger.info(f"Total processed: {samples_processed}/{total_rows}")
    logger.info(f"Preprocessed files saved to: {config.preprocessed_dir}")
