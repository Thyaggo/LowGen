import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import os
import yaml
import warnings
import wandb

from miditok import REMI, TokenizerConfig
from pathlib import Path
from model import Transformer
from dataset import LowDataset, collate_fn
from encodec import EncodecModel

from torch.utils.data import DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision("high")

def get_dataloader(config, tokenizer_model, tokenizer_midi):
    """
    Get the dataloader for the training and validation dataset
    
    Args:
        config (dict): The configuration dictionary
        tokenizer_model (EncodecModel): The tokenizer model
    """
    ds_raw = LowDataset(tokenizer_model, tokenizer_midi, **config["LowDataset"])

    train_ds, val_ds = random_split(ds_raw, [len(ds_raw)-1, 1])

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
    midi_config = TokenizerConfig()
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
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_token"])
    
    # Compile the model to make it faster on the inference
    try:
        model: Transformer = torch.compile(model)
    except:
        warnings.warn("The model couldn't be compiled, it will be slower on the inference")

    for epoch in range(config["epochs"]):
        # Set the model to train mode and empty the cache
        model.train()
        torch.cuda.empty_cache()
        
        # Iterate over the training dataset
        batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for i, data in enumerate(batch_iterator):
            
            # Get the input and label codes, if you choose flash_attention, you won't need the mask
            input_codes = data["input_codes"].to(DEVICE) # (B, T)
            label_input = data["label_input"].to(DEVICE) # (B, K, T)
            label_codes = data["label_codes"].to(DEVICE) # (B, K, T)
            
            optimizer.zero_grad()
            
            # Model forward
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                encoder_output = model.encode(input_codes)
                decode_output = model.decode(encoder_output, label_input)
                proj_output = model.project(decode_output)

                loss = criterion(proj_output.view(-1, proj_output.shape[-1]), label_codes.view(-1))
            
            #wandb.log({"loss": loss.item()})
            batch_iterator.set_postfix({"loss": loss.item()})
            
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        
        lr_scheduler.step()

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
        for i, batch in enumerate(val_dataloader):
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
                if label_input.size(2) >= config["sliding_window"]:
                    out = model.decode(src=input, tgt=label_input[:, :, -config["sliding_window"]:])
                else:
                    out = model.decode(src=input, tgt=label_input)

                # get next token
                prob = model.project(out[:, -1].unsqueeze(1))
                _, next_word = torch.max(prob, dim=-1)
                
                label_input = torch.cat(
                    [label_input, next_word], dim=2
                )

                if torch.any(next_word == config["eos_token"]):
                    break
                
                if label_input.size(2) % 100 == 0:
                    print("Iteration", label_input.size(2))
                    
            # Remove the bos and eos token
            mask = ((label_input == config["bos_token"]).any(dim=1) | (label_input ==config["eos_token"]).any(dim=1)).squeeze()
            label_input = label_input[:, :, ~mask]
            # Decode the discrete tokens to waveform
            wave = tokenizer_model.decode([(label_input, None)])
            # Save the waveform
            if not os.path.exists(config["output_dir"]):
                os.makedirs(config["output_dir"])
            output_path = config["output_dir"]
            torchaudio.save(f"{output_path}/output_{i}.wav", wave.cpu().squeeze(0), 24000)


if __name__ == "__main__":
    # Open the config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    #wandb.init(project="LowGen",config=config)
    
    train_model(config)