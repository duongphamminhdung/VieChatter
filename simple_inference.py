"""
Simple Inference Script for Chatterbox TTS
Loads fine-tuned model and generates audio from text.
"""

import os
import torch
import argparse
import soundfile as sf
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3
from src.utils import setup_logger


logger = setup_logger("SimpleInference")


def load_model(
    model_dir: str,
    checkpoint_path: str,
    is_turbo: bool = True,
    new_vocab_size: int = 52260,
    device: str = "cuda"
):
    """
    Load the fine-tuned Chatterbox model.

    Args:
        model_dir: Path to base pretrained models
        checkpoint_path: Path to fine-tuned checkpoint (.pt file)
        is_turbo: Whether to use Turbo mode
        new_vocab_size: Vocabulary size for the fine-tuned model
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded TTS engine ready for inference
    """
    logger.info(f"Loading {'TURBO' if is_turbo else 'NORMAL'} model...")
    logger.info(f"Base models from: {model_dir}")
    logger.info(f"Fine-tuned weights: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train a model first using train.py"
        )

    # Load base TTS engine
    EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS
    tts_engine = EngineClass.from_local(model_dir, device="cpu")

    # Initialize new T3 model with correct vocabulary size
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = new_vocab_size

    new_t3 = T3(hp=t3_config)

    # Remove wte layer for turbo mode if needed
    if is_turbo and hasattr(new_t3.tfmr, "wte"):
        logger.info("Turbo mode: Removing 'wte' layer from new T3 model")
        del new_t3.tfmr.wte

    # Load fine-tuned weights
    logger.info("Loading fine-tuned weights...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_t3.load_state_dict(state_dict, strict=True)

    # Replace T3 in the engine
    tts_engine.t3 = new_t3

    # Move to device and set to eval mode
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()

    tts_engine.device = device

    logger.info(f"✓ Model loaded successfully on {device}")
    return tts_engine


def inference(
    engine,
    text: str,
    prompt_audio_path: str,
    output_path: str,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    cfg_weight: float = 0.5,
    exaggeration: float = 0.5,
):
    """
    Generate audio from text using the loaded model.

    Args:
        engine: Loaded TTS engine
        text: Input text to synthesize
        prompt_audio_path: Path to reference audio for voice cloning
        output_path: Path to save output audio
        temperature: Sampling temperature (higher = more diverse)
        repetition_penalty: Penalty for repeating tokens
        cfg_weight: Classifier-free guidance weight (normal mode only)
        exaggeration: Exaggeration factor (turbo mode only)
    """
    logger.info("=" * 70)
    logger.info("INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Text: {text}")
    logger.info(f"Prompt audio: {prompt_audio_path}")
    logger.info(f"Output: {output_path}")

    # Check if prompt audio exists
    if not os.path.exists(prompt_audio_path):
        raise FileNotFoundError(f"Prompt audio not found: {prompt_audio_path}")

    # Generate parameters based on model type
    is_turbo = isinstance(engine, ChatterboxTurboTTS)

    if is_turbo:
        generation_params = {
            "temperature": temperature,
            "exaggeration": exaggeration,
            "repetition_penalty": repetition_penalty,
        }
        logger.info(f"Mode: TURBO (temp={temperature}, exaggeration={exaggeration})")
    else:
        generation_params = {
            "temperature": temperature,
            "cfg_weight": cfg_weight,
            "repetition_penalty": repetition_penalty,
        }
        logger.info(f"Mode: NORMAL (temp={temperature}, cfg_weight={cfg_weight})")

    # Generate audio
    logger.info("Generating audio...")
    with torch.no_grad():
        wav = engine.generate(
            text=text,
            audio_prompt_path=prompt_audio_path,
            **generation_params
        )

    # Save output
    wav_np = wav.squeeze().cpu().numpy()
    sf.write(output_path, wav_np, engine.sr)

    logger.info("=" * 70)
    logger.info("✓ Audio generated successfully!")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Sample rate: {engine.sr} Hz")
    logger.info(f"Duration: {len(wav_np) / engine.sr:.2f} seconds")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Simple inference script for Chatterbox TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to reference audio file for voice cloning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path to save output audio file"
    )

    # Model configuration
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./pretrained_models",
        help="Path to base pretrained models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        default=True,
        help="Use Turbo mode (default: True)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=52260,
        help="Vocabulary size of fine-tuned model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.1-1.5)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (1.0-2.0)"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight for normal mode (0.0-1.0)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration factor for turbo mode (0.0-1.0)"
    )

    args = parser.parse_args()

    # Load model
    engine = load_model(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        is_turbo=args.turbo,
        new_vocab_size=args.vocab_size,
        device=args.device
    )

    # Run inference
    inference(
        engine=engine,
        text=args.text,
        prompt_audio_path=args.prompt,
        output_path=args.output,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration
    )


if __name__ == "__main__":
    main()

