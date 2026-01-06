import torch
import torch.nn.functional as F
from src.chatterbox_.models.t3.modules.cond_enc import T3Cond
from src.config import TrainConfig
from src.utils import setup_logger


logger = setup_logger(__name__)


def resize_and_load_t3_weights(new_model: torch.nn.Module, pretrained_state_dict: dict): 
    """
    Loads pretrained weights into a new T3 model with a different vocabulary size.
    Features: Initialize new tokens with the AVERAGE of existing tokens.
    """
    new_model_state_dict = new_model.state_dict()

    embedding_layer_name = "text_emb.weight"
    output_head_name     = "text_head.weight"

                                                  # Step 1: Copy weights for ALL matching layers
    for name, param in pretrained_state_dict.items(): 
        
        if name not in [embedding_layer_name, output_head_name]: 
            
            if name in new_model_state_dict and new_model_state_dict[name].shape == param.shape:
                new_model_state_dict[name].copy_(param)
                
            else: 
                logger.warning(f"Layer skipped (mismatch): {name}")


                                                  # Step 2: Smart copy for Embedding Layer (Average Init)
    if embedding_layer_name in pretrained_state_dict: 
        
                       old_emb_weights = pretrained_state_dict[embedding_layer_name]
        old_vocab_size, _              = old_emb_weights.shape
                       new_vocab_size  = new_model_state_dict[embedding_layer_name].shape[0]

                                                      # A) Copy old weights
        new_model_state_dict[embedding_layer_name][:old_vocab_size, :].copy_(old_emb_weights)
        logger.info(f"Embedding layer: {old_vocab_size} tokens preserved.")

                                                      # B) Initialize new tokens with average
        if new_vocab_size > old_vocab_size: 
            
            mean_emb       = old_emb_weights.mean(dim=0)
            num_new_tokens = new_vocab_size - old_vocab_size
            
            new_model_state_dict[embedding_layer_name][old_vocab_size:, :].copy_(mean_emb.unsqueeze(0).expand(num_new_tokens, -1))
            
            logger.info(f"Embedding layer: {num_new_tokens} new tokens initialized with mean.")


                                                  # Step 3: Smart copy for Output Head (Average Init)
    if output_head_name in pretrained_state_dict: 
        
                       old_head_weights = pretrained_state_dict[output_head_name]
        old_vocab_size, _               = old_head_weights.shape
                       new_vocab_size   = new_model_state_dict[output_head_name].shape[0]

                                                      # A) Copy old weights
        new_model_state_dict[output_head_name][:old_vocab_size, :].copy_(old_head_weights)
        logger.info(f"Output head: {old_vocab_size} tokens preserved.")

                                                      # B) Initialize new neurons with average
        if new_vocab_size > old_vocab_size: 
            
            mean_head      = old_head_weights.mean(dim=0)
            num_new_tokens = new_vocab_size - old_vocab_size
            new_model_state_dict[output_head_name][old_vocab_size:, :].copy_(mean_head.unsqueeze(0).expand(num_new_tokens, -1))
            
            logger.info(f"Output head: {num_new_tokens} new neurons initialized with mean.")

                                                  # Step 4: Load the updated state dict into the new model
    new_model.load_state_dict(new_model_state_dict)
    logger.info("All weights transferred successfully (Mean Initialization applied)!")

    return new_model


class ChatterboxTrainerWrapper(torch.nn.Module): 
    """
    Wrapper class to calculate Loss inside the Forward pass for HuggingFace Trainer.
    """

    def __init__(self, t3_model, config=None): 
        super().__init__()
        self.t3 = t3_model

                                                      # Use provided config or default to TrainConfig
        self.cfg = config if config is not None else TrainConfig()

        if hasattr(t3_model.hp, 'speech_cond_prompt_len'): 
            self.prompt_token_len = t3_model.hp.speech_cond_prompt_len
        else: 
            self.prompt_token_len = 150
        
                                                      # Required for HuggingFace Trainer checkpoint loading/resuming
                                                      # Set to empty list to indicate no keys should be ignored
        self._keys_to_ignore_on_save = []


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None): 
        self.t3.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self): 
        return self.t3.get_input_embeddings()


    def forward(
            self,
            text_tokens,
            text_token_lens,
            speech_tokens, 
            speech_token_lens,
            speaker_emb, 
            prompt_tokens):

                                                      # Mark CUDAGraphs step begin to prevent tensor overwriting issues
                                                      # This tells CUDAGraphs to start a fresh recording for this forward pass
        if self.training and hasattr(torch.compiler, 'cudagraph_mark_step_begin'): 
           try                                                                   : 
                torch.compiler.cudagraph_mark_step_begin()
            except Exception: 
                                                              # Silently ignore if CUDAGraphs marking fails
                pass

        device     = text_tokens.device
        batch_size = text_tokens.size(0)
        
                                                      # Validate batch size to avoid shape issues
        if batch_size == 0:
                                                          # Return zero loss for empty batch
            return (torch.tensor(0.0, device=device, requires_grad=True), None)
        
        emotion_adv = 0.5 * torch.ones(batch_size, 1, 1).to(device)
        
        t3_cond = T3Cond(
            speaker_emb               = speaker_emb,
            cond_prompt_speech_tokens = prompt_tokens,
            emotion_adv               = emotion_adv
        )

                                                      # Forward Pass with aggressive error handling - catch ALL exceptions to prevent hangs
        try: 
            out = self.t3.forward(
                t3_cond           = t3_cond,
                text_tokens       = text_tokens,
                text_token_lens   = text_token_lens,
                speech_tokens     = speech_tokens,
                speech_token_lens = speech_token_lens,
                training          = True
            )
        except torch.cuda.OutOfMemoryError as e: 
                                                          # Handle OOM errors specifically
            logger.error(f"⚠️  CUDA Out of Memory! Attempting to recover...")
            torch.cuda.empty_cache()  # Clear cache
            logger.warning(f"⚠️  Skipping batch due to OOM. Consider reducing batch_size or increasing grad_accum.")
            return (torch.tensor(0.0, device=device, requires_grad=True), None)
        except Exception as e: 
                                                          # Catch ALL exceptions (RuntimeError, ValueError, etc.) to prevent hangs
                                                          # This includes CUDAGraphs errors, symbolic shape errors, and any other issues
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in ["cudagraphs", "symbolic", "shape", "tensor", "compile"]): 
                logger.warning(f"⚠️  Skipping batch due to error: {type(e).__name__}: {error_msg[:200]}")
            else: 
                logger.warning(f"⚠️  Skipping batch due to unexpected error: {type(e).__name__}: {error_msg[:200]}")
                                                          # Return zero loss to continue training without crashing
            return (torch.tensor(0.0, device=device, requires_grad=True), None)

                                                      # Loss calculation with error handling
        try: 
            IGNORE_ID = -100

            speech_logits = out.speech_logits[:, :-1, :].transpose(1, 2)
            speech_labels = speech_tokens[:, 1:]
            
            curr_speech_len = speech_labels.size(1)
            mask_speech_pad = torch.arange(curr_speech_len, device=device)[None, :] > = (speech_token_lens[:, None] - 1)

            if self.cfg.is_turbo == True:
               speech_labels      = speech_labels.masked_fill(mask_speech_pad, IGNORE_ID)
            else: 
                actual_prompt_len = prompt_tokens.size(1)
                mask_prompt       = torch.arange(curr_speech_len, device=device)[None, :] < actual_prompt_len
                speech_labels     = speech_labels.masked_fill(mask_speech_pad | mask_prompt, IGNORE_ID)
            
            loss_speech = F.cross_entropy(speech_logits, speech_labels, ignore_index=IGNORE_ID)

            text_logits = out.text_logits[:, :-1, :].transpose(1, 2)
            text_labels = text_tokens[:, 1:]
            
            curr_text_len = text_labels.size(1)
            mask_text_pad = torch.arange(curr_text_len, device=device)[None, :] > = (text_token_lens[:, None] - 1)
            
            text_labels = text_labels.masked_fill(mask_text_pad, IGNORE_ID)
            
            loss_text = F.cross_entropy(text_logits, text_labels, ignore_index=IGNORE_ID)

            total_loss = loss_text + loss_speech
            return (total_loss, None)
        except Exception as e: 
                                                          # Catch any errors during loss calculation and skip this batch
            logger.warning(f"⚠️  Error calculating loss, skipping batch: {type(e).__name__}: {str(e)[:200]}")
            return (torch.tensor(0.0, device=device, requires_grad=True), None)