"""
Dataset class that reads directly from parquet files and processes audio on-the-fly during training.
Eliminates the need for preprocessing step - saves time, storage, and uses GPU efficiently.
"""

import os
import torch
import torchaudio
import dask.dataframe as dd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger


logger = setup_logger(__name__)


class ParquetDataset(Dataset):
    """
    Dataset that streams parquet files and processes audio on-the-fly.
    
    Benefits:
    - No preprocessing step (saves hours/days)
    - No duplicate storage (~45 GB saved)
    - Uses GPU for embeddings/tokenization (faster)
    - Memory efficient (only partition loaded at a time)
    - Can train immediately after setup
    """
    
    def __init__(self, config, tts_engine, split="train", num_samples=None):
        self.cfg = config
        self.parquet_path = config.parquet_path
        self.tts_engine = tts_engine
        self.split = split
        
        # Validate parquet path
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"Parquet directory does not exist: {self.parquet_path}"
            )
        
        logger.info(f"Loading parquet files from: {self.parquet_path}")
        logger.info(f"Using split: {split}")

        # Read parquet with Dask (lazy loading, no RAM usage)
        self.df = dd.read_parquet(self.parquet_path)
        self.npartitions = self.df.npartitions
        logger.info(f"Total partitions: {self.npartitions}")

        # Get total number of rows (may take a moment)
        total_rows = len(self.df)
        logger.info(f"Total rows in dataset: {total_rows}")

        # Limit samples if specified
        self.num_samples = num_samples
        if num_samples is not None and num_samples < total_rows:
            logger.info(f"Limiting to {num_samples} samples")
            self.df = self.df.head(num_samples, compute=True)
            # Convert back to Dask with single partition
            self.df = dd.from_pandas(self.df, npartitions=1)
            self.npartitions = 1
            logger.info(f"Using subset: {num_samples} rows (single partition)")
        else:
            logger.info(f"Using full dataset: {total_rows} rows")

        # Precompute partition boundaries for O(1) lookup
        logger.info("Precomputing partition boundaries...")
        self.partition_start_indices = [0]
        cumulative = 0
        for p_idx in range(self.npartitions):
            partition = self.df.get_partition(p_idx)
            partition_len = len(partition)
            cumulative += partition_len
            self.partition_start_indices.append(cumulative)
        logger.info(f"Partition boundaries computed: {len(self.partition_start_indices) - 1} partitions")

        self.current_partition = -1
        self.partition_df = None
        self.partition_length = 0
        self.partition_start_idx = 0
        
        # Prepare resampler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=S3_SR
        )
        
        # Get tokens
        self.sot_token = config.start_text_token
        self.eot_token = config.stop_text_token
        
        # Move TTS to correct device
        self.tts_engine.ve.to(self.device)
        self.tts_engine.s3gen.to(self.device)
        
        logger.info("Dataset initialized (ready for training without preprocessing)")
    
    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.df)
    
    def load_partition_for_idx(self, global_idx):
        """Load the partition that contains the global index."""
        # For single partition case (when num_samples is set)
        if self.npartitions == 1:
            if self.partition_df is None:
                self.partition_df = self.df.get_partition(0).compute()
                self.partition_length = len(self.partition_df)
                self.partition_start_idx = 0
            return

        # O(1) lookup using binary search on precomputed boundaries
        import bisect
        partition_idx = bisect.bisect_right(self.partition_start_indices, global_idx) - 1
        partition_start = self.partition_start_indices[partition_idx]

        # Load partition if needed
        if self.partition_df is None or partition_idx != self.current_partition:
            logger.info(f"Loading partition {partition_idx + 1}/{self.npartitions}")
            self.partition_df = self.df.get_partition(partition_idx).compute()
            self.partition_length = len(self.partition_df)
            self.current_partition = partition_idx
            self.partition_start_idx = partition_start
    
    def process_audio(self, audio_data):
        """Process audio data on-the-fly (uses GPU)."""
        
        if audio_data is None:
            raise ValueError("Audio data is None")
        
        # Load audio from different formats
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            # File path
            wav, sr = torchaudio.load(audio_data)
        elif hasattr(audio_data, 'array'):
            # Embedded audio with .array attribute
            audio_array = np.array(audio_data.array)
            wav = torch.from_numpy(audio_array).unsqueeze(0).float()
            sr = audio_data.sampling_rate if hasattr(audio_data, 'sampling_rate') else 24000
        elif isinstance(audio_data, dict):
            import io
            if 'array' in audio_data:
                # Dict with array
                audio_array = np.array(audio_data['array'])
                wav = torch.from_numpy(audio_array).unsqueeze(0).float()
                sr = audio_data.get('sampling_rate', 24000)
            elif 'bytes' in audio_data:
                # Raw WAV bytes (PhoAudiobook format)
                wav_bytes = audio_data['bytes']
                wav_io = io.BytesIO(wav_bytes)
                wav, sr = torchaudio.load(wav_io)
            else:
                raise ValueError(f"Unsupported audio format: {type(audio_data)}")
        else:
            raise ValueError(f"Invalid audio data type: {type(audio_data)}")
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != S3_SR:
            wav = self.resampler(wav)
        
        # Move to device
        wav = wav.to(self.device)
        
        # Skip if too short
        if wav.shape[1] < 0.5 * S3_SR:  # Less than 0.5 seconds
            raise ValueError(f"Audio too short: {wav.shape[1]} samples (< 0.5 seconds)")
        
        return wav, sr
    
    def process_row(self, partition_df, idx):
        """Process a single row: extract features on-the-fly."""
        row = partition_df.iloc[idx]
        
        # Get text
        text = row.get('text', '')
        if not text or str(text).strip() == '':
            raise ValueError(f"Empty text at index {idx}")
        
        # Process audio (on-the-fly, GPU accelerated)
        audio_data = row.get('audio')
        wav, sr = self.process_audio(audio_data)
        
        # Extract speaker embedding (GPU accelerated)
        with torch.no_grad():
            wav_np = wav.cpu().squeeze().numpy()
            spk_emb_np = self.tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
            speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()
        
        # Tokenize full speech (GPU accelerated)
        with torch.no_grad():
            s_tokens, _ = self.tts_engine.s3gen.tokenizer(wav.unsqueeze(0).to(self.device))
            speech_tokens = s_tokens.squeeze().cpu()
        
        # Extract prompt (first N seconds)
        prompt_samples = int(self.cfg.prompt_duration * S3_SR)
        if wav.shape[1] < prompt_samples:
            prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
        else:
            prompt_wav = wav[:, :prompt_samples]
        
        with torch.no_grad():
            p_tokens, _ = self.tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0).to(self.device))
            prompt_tokens = p_tokens.squeeze().cpu()
        
        # Clean and normalize text
        raw_text = str(text).strip()
        from src.chatterbox_.tts import punc_norm
        clean_text = punc_norm(raw_text)

        # Tokenize text using the engine's tokenizer
        # Handle both custom EnTokenizer and HuggingFace GPT2TokenizerFast
        if hasattr(self.tts_engine.tokenizer, 'text_to_tokens'):
            # Custom tokenizer (EnTokenizer or MTLTokenizer)
            text_tokens = self.tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()
        else:
            # HuggingFace tokenizer (GPT2TokenizerFast)
            token_output = self.tts_engine.tokenizer(clean_text, return_tensors="pt")
            text_tokens = token_output.input_ids[0].cpu()
        
        # Add special tokens
        sot = torch.tensor([self.sot_token], dtype=torch.long).cpu()
        eot = torch.tensor([self.eot_token], dtype=torch.long).cpu()
        text_tokens = torch.cat([sot, text_tokens, eot])
        
        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_emb": speaker_emb,
            "prompt_tokens": prompt_tokens,
        }
    
    def __getitem__(self, idx):
        """Get a sample (processes audio on-the-fly with GPU)."""

        # Load the correct partition for this index
        self.load_partition_for_idx(idx)

        # Calculate local index within the current partition
        if self.npartitions == 1:
            local_idx = idx
        else:
            local_idx = idx - self.partition_start_idx

        # Process row (audio extraction + feature generation)
        return self.process_row(self.partition_df, local_idx)
    
    def process_partition_for_dataloader(self, partition_idx, indices):
        """
        Process a specific partition and return samples for DataLoader.
        This is used by the DataLoader to batch samples.
        """
        logger.info(f"Processing partition {partition_idx} for DataLoader...")
        self.current_partition = partition_idx
        self.partition_df = self.df.get_partition(partition_idx).compute()
        self.partition_length = len(self.partition_df)
        
        samples = []
        for idx in tqdm(indices, desc=f"Loading partition {partition_idx}"):
            try:
                sample = self.process_row(self.partition_df, idx)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                # Add None to maintain batch alignment
                samples.append(None)
        
        return samples


def data_collator_parquet(batch):
    """Collator for ParquetDataset."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    text_tokens = pad_sequence(
        [x["text_tokens"] for x in batch],
        batch_first=True,
        padding_value=0
    )
    speech_tokens = pad_sequence(
        [x["speech_tokens"] for x in batch],
        batch_first=True,
        padding_value=0
    )
    prompt_tokens = pad_sequence(
        [x["prompt_tokens"] for x in batch],
        batch_first=True,
        padding_value=0
    )

    # Stack speaker embeddings
    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Calculate token lengths (actual lengths before padding)
    text_token_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_token_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_token_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
    }


class SequentialPartitionDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset that processes partitions sequentially.
    - Loads one partition at a time
    - Uses workers to process samples within that partition
    - Then moves to next partition
    """

    def __init__(self, parquet_dataset):
        self.dataset = parquet_dataset
        self.current_partition = 0
        self.partition_data = None

    def __len__(self):
        """Return total number of samples across all partitions."""
        return len(self.dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker mode - process all partitions
            for partition_idx in range(self.dataset.npartitions):
                self._load_partition(partition_idx)
                for idx in range(len(self.partition_data)):
                    yield self._process_row(idx)
        else:
            # Multi-worker mode - each worker gets different partitions
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            for partition_idx in range(worker_id, self.dataset.npartitions, num_workers):
                self._load_partition(partition_idx)
                for idx in range(len(self.partition_data)):
                    yield self._process_row(idx)

    def _load_partition(self, partition_idx):
        """Load a single partition."""
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'main'
        logger.info(f"[Worker {worker_id}] Loading partition {partition_idx + 1}/{self.dataset.npartitions}")
        self.partition_data = self.dataset.df.get_partition(partition_idx).compute()
        self.current_partition = partition_idx

    def _process_row(self, local_idx):
        """Process a row from current partition."""
        try:
            return self.dataset.process_row(self.partition_data, local_idx)
        except Exception as e:
            logger.error(f"Error processing sample in partition {self.current_partition}, idx {local_idx}: {e}")
            return None


def create_sequential_parquet_dataloader(config, tts_engine, split="train", num_samples=None, batch_size=8, num_workers=4):
    """
    Create a DataLoader that processes partitions sequentially.
    - Each partition is fully loaded
    - Workers process samples within the partition in parallel
    - Then moves to next partition
    """
    dataset = ParquetDataset(config, tts_engine, split=split, num_samples=num_samples)

    # Create sequential iterable dataset
    sequential_dataset = SequentialPartitionDataset(dataset)

    # Create DataLoader with multiple workers
    dataloader = torch.utils.data.DataLoader(
        sequential_dataset,
        batch_size  = batch_size,
        collate_fn  = data_collator_parquet,
        num_workers = num_workers,
        pin_memory  = True
    )

    logger.info(f"Created Sequential DataLoader with {num_workers} workers")
    logger.info(f"Processing {dataset.npartitions} partitions sequentially")
    logger.info(f"Total samples: {len(dataloader.dataset.dataset)}")

    return dataloader, dataset

