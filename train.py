import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import yaml

from model import Transformer
from dataset import LowDataset
from encodec import EncodecModel

from torch.utils.data import DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_dataloader(config, tokenizer_model):
    # It only has the train split, so we divide it overselves
    # Keep 90% for training, 10% for validation

    ds_raw = LowDataset(tokenizer_model, **config["LowDataset"])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds, val_ds = random_split(ds_raw, [469, 1])

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader

def encodec_model(stereo: bool = False, bandwidth: float = 6.0, device: str = "cpu"):
    if stereo:
        model = EncodecModel.encodec_model_48khz().to(device)
    else:
        model = EncodecModel.encodec_model_24khz().to(device)

    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.

    assert bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0], "Invalid bandwidth"

    model.set_target_bandwidth(bandwidth)
    return model

def train_model(config):
    tokenizer_model = encodec_model(config["stereo"], config["bandwidth"], DEVICE)

    train_dataloader, val_dataloader = get_dataloader(config, tokenizer_model)

    model = Transformer(**config["Transformer"]).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_token"])
    
    model: Transformer = torch.compile(model)

    for epoch in range(config["epochs"]):
        model.train()
        torch.cuda.empty_cache()
        batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for i, data in enumerate(batch_iterator):

            input_codes = data["input_codes"].to(DEVICE) # (B, K, T)
            #input_mask = data["input_mask"].to(DEVICE) # (B, T)
            label_input = data["label_input"].to(DEVICE) # (B, K, T)
            label_codes = data["label_codes"].to(DEVICE) # (B, K, T)
            #decoder_mask = data["label_mask"].to(DEVICE) # (B, T) & (B, T, T)
            
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
    model.eval()

    with torch.no_grad():
        for batch in val_dataloader:
            input_codes = batch["input_codes"].to(DEVICE) # (B, K, T)
            #input_mask = batch["input_mask"].to(DEVICE) # (B, T)

            # check that the batch size is 1
            assert input_codes.size(0) == 1, "Batch size must be 1 for validation"
            
            input = model.encode(input_codes)
            
            # Initialize the decoder input with the sos token
            label_input = torch.empty(1, config["codebook_num"], 1).fill_(config["bos_token"]).type_as(input_codes).to(DEVICE)
            while True:
                if label_input.size(2) == 10:
                    break

                # # build mask for target
                # tgt_mask = model.generate_square_subsequent_mask(label_input.size(2)).type_as(input_mask).to(DEVICE)

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

        wave = tokenizer_model.decode([(label_input, None)])
        torchaudio.save(f"output.wav", wave, 24000)


if __name__ == "__main__":
    train_model(config)