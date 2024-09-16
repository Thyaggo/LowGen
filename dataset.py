import os
import pandas as pd
import torchaudio
import torch
import yaml
from typing import Dict, Callable
from encodec.utils import convert_audio
from miditok import REMI, TokenizerConfig

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

class LowDataset(Dataset):
    def __init__(self, 
                 tokenizer_model,
                 tokenizer_midi,
                 stereo: bool = False,
                 pad_token: int = 1025,
                 bos_token: int = 1024,
                 eos_token: int = 1026,
                 masking: bool = False,
                 dir_labels: str = "data/lowdata",
                 dir_midi: str = "data/lowmidi"
                 ):
        super().__init__()
        
        self.data_path = os.listdir(dir_labels)
        self.dir_labels = dir_labels
        self.dir_midi = dir_midi
            
        self.device = device
        self.tokenizer = tokenizer_midi
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.masking = masking
        self.dir_labels = dir_labels
        self.channels = 2 if stereo else 1
        self.sample_rate = 48000 if stereo else 24000

        self.model = tokenizer_model

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        label_path = f"{self.dir_labels}/{self.data_path[idx]}"
        midi_path = f"{self.dir_midi}/{self.data_path[idx][:-4]}.mid"
        
        label_wav, label_sr = torchaudio.load(label_path)
        
        input_codes = self.tokenizer.encode(midi_path)[0].ids
        
        label_wav = convert_audio(label_wav, label_sr, self.sample_rate, self.channels)
        
        with torch.no_grad():
            frames = self.model.encode(label_wav.unsqueeze(0).to(self.device))
        label_codes = torch.cat([encoded[0] for encoded in frames], dim=-1).squeeze(0)
        
        input_codes = torch.cat([torch.tensor([self.tokenizer["BOS_None"]]), 
                                 torch.tensor(input_codes),
                                 torch.tensor([self.tokenizer["EOS_None"]])], dim=-1)
        
        label_input = torch.cat([torch.empty((label_codes.size(0), 1), device=device, dtype=label_codes.dtype).fill_(self.bos_token),
                                label_codes], dim=-1)
        
        label_codes = torch.cat([label_codes, 
                                 torch.empty((label_codes.size(0), 1), device=device, dtype=label_codes.dtype).fill_(self.eos_token)], dim=-1)

        return {
            "input_codes": input_codes,
            "label_input": label_input,
            "label_codes": label_codes
        }
        
    
        

def collate_fn(midi_pad: int, model_pad: int) -> Callable:  
    """
    Custom collate function to pad sequences in a batch.
    
    Args:
        batch: List of samples from the dataset.
        midi_pad: Padding value for MIDI sequences.
        model_pad: Padding value for model sequences.
        
    Returns:
        A dictionary containing padded input codes, label input, and label codes.
    """
    def collate(batch, midi_pad = midi_pad, model_pad = model_pad) -> dict:
        input_codes = pad_sequence([item["input_codes"] for item in batch], batch_first=True, padding_value=midi_pad)
        label_input = pad_sequence([item["label_input"] for item in batch], batch_first=True, padding_value=model_pad)
        label_codes = pad_sequence([item["label_codes"] for item in batch], batch_first=True, padding_value=model_pad)
        
        return {
            "input_codes": input_codes,
            "label_input": label_input,
            "label_codes": label_codes
        }
    return collate



def test(config):
    from encodec import EncodecModel
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    tokenizer_model.set_target_bandwidth(config["bandwidth"])
    
    midi_config = TokenizerConfig()
    tokenizer_midi = REMI(midi_config)
    
    dataset = LowDataset(tokenizer_model, tokenizer_midi, **config["LowDataset"])

    dataset[0]
    print("Test passed!")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    test(config)