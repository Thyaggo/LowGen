import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import yaml

from miditok import REMI, TokenizerConfig
from pathlib import Path
from model import Transformer
from dataset import LowDataset, collate_fn
from encodec import EncodecModel

from torch.utils.data import DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloader(config, tokenizer_model, tokenizer_midi):
    """
    Get the dataloader for the training and validation dataset
    
    Args:
        config (dict): The configuration dictionary
        tokenizer_model (EncodecModel): The tokenizer model
    """
    ds_raw = LowDataset(tokenizer_model, **config["LowDataset"])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds, val_ds = random_split(ds_raw, [469, 1])

    collate = collate_fn(tokenizer_midi.pad_token_id, config["pad_token"])
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader

def encodec_model(stereo: bool = False, bandwidth: float = 6.0, device: str = "cpu"):
    """
    Get the EncodecModel for become waveforms to discrete tokens
    
    Args:
        stereo (bool): Speficy the channels of the audio and the model parameters.
        bandwidth (float): The target bandwidth choose the number of codebooks, 1.5kbps (n_q = 2),
        3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    
    Returns:
        EncodecModel
    """
    
    model = EncodecModel.encodec_model_24khz().to(device) if not stereo else EncodecModel.encodec_model_48khz().to(device)

    assert bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0], "Invalid bandwidth"

    model.set_target_bandwidth(bandwidth)
    return model

def midi_tokenizer(config):
    """Train the tokenizer model on the midi files
    """
    midi_config = TokenizerConfig(**config["TokenizerConfig"])
    tokenizer = REMI(midi_config)
    midi_paths = list(Path(config["LowDataset"]["dir_midi"]).glob("**/*.mid"))
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    return tokenizer

def train_model(config):
    # Get the tokenizer model
    tokenizer_model = encodec_model(config["stereo"], config["bandwidth"], DEVICE)
    # Get the midi tokenizer
    tokenizer_midi = midi_tokenizer(config)
    # Get the dataloader
    train_dataloader, val_dataloader = get_dataloader(config, tokenizer_model, tokenizer_midi)
    # Get the model
    model = Transformer(pad_midi = tokenizer_midi.pad_token_id, **config["Transformer"]).to(DEVICE)

    # Initialize the optimizer and the criterion
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_token"])
    
    # Compile the model to make it faster on the inference
    try:
        model: Transformer = torch.compile(model)
    except:
        print("Compilation failed, the model will be slower on the inference")

    for epoch in range(config["epochs"]):
        # Set the model to train mode and empty the cache
        model.train()
        torch.cuda.empty_cache()
        
        # Iterate over the training dataset
        batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for i, data in enumerate(batch_iterator):
            
            # Get the input and label codes, if you choose flash_attention, you won't need the mask
            input_codes = data["input_codes"].to(DEVICE) # (B, K, T)
            #input_mask = data["input_mask"].to(DEVICE) # (B, T)
            label_input = data["label_input"].to(DEVICE) # (B, K, T)
            label_codes = data["label_codes"].to(DEVICE) # (B, K, T)
            #decoder_mask = data["label_mask"].to(DEVICE) # (B, T) & (B, T, T)
            
            # Model forward
            encoder_output = model.encode(input_codes)
            decode_output = model.decode(encoder_output, label_input)
            proj_output = model.project(decode_output)

            loss = criterion(proj_output.view(-1, proj_output.shape[-1]), label_codes.view(-1))
            
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")

    run_validation(config, model, tokenizer_model, val_dataloader)

def run_validation(config: dict, model: Transformer, tokenizer_model: EncodecModel, val_dataloader: DataLoader):
    """
    Run the validation on the model and save the output waveform

    Args:
        config (dict): The configuration dictionary
        model (Transformer): The transformer model
        tokenizer_model (EncodecModel): The tokenizer model
        val_dataloader (DataLoader): The validation dataloader
    """
    model.eval()

    with torch.no_grad():
        for batch in val_dataloader:
            input_codes = batch["input_codes"].to(DEVICE) # (B, K, T)

            # check that the batch size is 1
            assert input_codes.size(0) == 1, "Batch size must be 1 for validation"
            
            input = model.encode(input_codes)
            
            # Initialize the decoder input with the bos token
            label_input = torch.empty(1, config["codebook_num"], 1).fill_(config["bos_token"]).type_as(input_codes).to(DEVICE)
            while True:
                if label_input.size(2) == config["max_len_token"]:
                    break

                # calculate output
                out = model.decode(src=input, tgt=label_input)

                # get next token
                prob = model.project(out[:, -1].unsqueeze(1))
                _, next_word = torch.max(prob, dim=-1)
                
                label_input = torch.cat(
                    [label_input, next_word], dim=2
                )

                if torch.any(next_word == config["eos_token"]):
                    break
        # Remove the bos and eos token
        mask = ((label_input == config["bos_token"]).any(dim=1) | (label_input ==config["eos_token"]).any(dim=1)).squeeze()
        label_input = label_input[:, :, ~mask]
        # Decode the discrete tokens to waveform
        wave = tokenizer_model.decode([(label_input, None)])
        # Save the waveform
        torchaudio.save(f"output.wav", wave.cpu().squeeze(0), 24000)


if __name__ == "__main__":
    # Open the config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_model(config)