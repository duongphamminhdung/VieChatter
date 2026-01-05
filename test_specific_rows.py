"""
Quick test to check randomly selected rows in the dataset.
This helps identify problematic audio data patterns.
"""

import dask.dataframe as dd
import os
import random
import io

# Read the dataset
parquet_path = "/content/drive/MyDrive/datasets/phoaudiobook/phoaudiobook"
splits = {'train': 'train-*.parquet', 'validation': 'validation-*.parquet', 'test': 'test-*.parquet'}

# Configuration
NUM_RANDOM_ROWS = 7  # Number of random rows to test
SPLIT = "train"  # Which split to test

print(f"Reading dataset from: {parquet_path}")
print(f"Using split: {SPLIT}")
print()

df = dd.read_parquet(os.path.join(parquet_path, splits[SPLIT]))

total_rows = len(df)
print(f"Total rows in {SPLIT} split: {total_rows}")
print(f"Number of partitions: {df.npartitions}")

# Generate random row indices
random.seed(42)  # For reproducibility
random_indices = sorted(random.sample(range(total_rows), min(NUM_RANDOM_ROWS, total_rows)))
print(f"\nTesting {len(random_indices)} randomly selected rows: {random_indices}")

# Determine which partitions to fetch
rows_per_partition = total_rows // df.npartitions
partitions_needed = set(idx // rows_per_partition for idx in random_indices if rows_per_partition > 0)
partitions_needed = sorted(partitions_needed)
print(f"Will fetch {len(partitions_needed)} partition(s): {partitions_needed}")

# Load needed partitions
partition_data = {}
for partition_idx in partitions_needed:
    print(f"\nFetching partition {partition_idx}...")
    partition_data[partition_idx] = df.get_partition(partition_idx).compute()

# Test the audio parsing logic
print(f"\n{'='*80}")
print(f"Testing audio parsing for {len(random_indices)} random rows:")
print(f"{'='*80}")

success_count = 0
failed_count = 0

for global_idx in random_indices:
    # Determine which partition this row is in
    partition_idx = global_idx // rows_per_partition if rows_per_partition > 0 else 0

    if partition_idx not in partition_data:
        print(f"\nRow {global_idx}: Partition {partition_idx} not loaded")
        failed_count += 1
        continue

    # Get local index within partition
    partition_df = partition_data[partition_idx]
    local_idx = global_idx % rows_per_partition if rows_per_partition > 0 else global_idx

    if local_idx >= len(partition_df):
        print(f"\nRow {global_idx}: NOT FOUND in partition {partition_idx}")
        failed_count += 1
        continue

    row = partition_df.iloc[local_idx]
    audio_data = row.get('audio')
    text = row.get('text', '')

    print(f"\n--- Row {global_idx} (Partition {partition_idx}, Local {local_idx}) ---")
    print(f"Text: {str(text)[:80]}")

    # Try to parse audio
    success = False
    error_msg = ""

       try               : 
    if audio_data is None: 
            error_msg = "Audio data is None"
        elif isinstance(audio_data, str):
            if os.path.exists(audio_data):
                print(f"✓ Audio is a valid file path: {audio_data}")
                success = True
            else:
                error_msg = f"Audio is a string but file doesn't exist: {audio_data[:100]}"
        elif hasattr(audio_data, 'array'):
            import numpy as np
            audio_array = np.array(audio_data.array)
            print(f"✓ Audio is embedded with array shape: {audio_array.shape}")
            print(f"  Sampling rate: {audio_data.sampling_rate if hasattr(audio_data, 'sampling_rate') else 'N/A'}")
            print(f"  Array dtype: {audio_array.dtype}")
            success = True
        elif isinstance(audio_data, dict):
            if 'array' in audio_data:
                import numpy as np
                print(f"✓ Audio is dict with keys: {list(audio_data.keys())}")
                audio_array = np.array(audio_data['array'])
                print(f"  Array shape: {audio_array.shape}")
                print(f"  Sampling rate: {audio_data.get('sampling_rate', 'N/A')}")
                success = True
            elif 'bytes' in audio_data:
                # Handle raw WAV bytes
                import torchaudio
                import numpy as np
                print(f"✓ Audio is dict with 'bytes' key (raw WAV data)")
                wav_bytes = audio_data['bytes']
                print(f"  Bytes length: {len(wav_bytes)}")

                # Convert bytes to WAV using torchaudio
                wav_io = io.BytesIO(wav_bytes)
                wav, sr = torchaudio.load(wav_io)
                print(f"  Loaded waveform shape: {wav.shape}, sample rate: {sr}")
                success = True
            else:
                error_msg = f"Audio is dict but no 'array' or 'bytes' key. Keys: {list(audio_data.keys())}"
        else:
            error_msg = f"Could not parse audio data at row {global_idx}. Type: {type(audio_data)}"
            error_msg += f" - Str: {str(audio_data)[:200]}"
    except Exception as e:
        error_msg = f"Exception while parsing: {e}"

    if success:
        print(f"✓ SUCCESS: Audio can be parsed")
        success_count += 1
    else:
        print(f"✗ FAILED: {error_msg}")
        failed_count += 1

print(f"\n{'='*80}")
print(f"Test complete!")
print(f"Results: {success_count} successful, {failed_count} failed out of {len(random_indices)} tests")
print(f"Success rate: {100 * success_count / len(random_indices):.1f}%")
print(f"{'='*80}")
