import os
import pandas as pd
import torchaudio
import torch
import yaml
from typing import Dict, Callable
from encodec.utils import convert_audio

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LowDataset(Dataset):
    def __init__(self, 
                 tokenizer_model,
                 tokenizer_midi,
                 max_duration: int,
                 stereo: bool = False,
                 pad_token: int = 1025,
                 bos_token: int = 1024,
                 eos_token: int = 1026,
                 masking: bool = False,
                 dir_inputs: str = "data/inputs",
                 dir_labels: str = "data/labels",
                 dir_midi: str = "data/out"
                 ):
        super().__init__()
        
        self.data_path = os.listdir(dir_labels)
        self.dir_midi = dir_midi
            
        self.device = device
        self.tokenizer = tokenizer_midi
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.masking = masking
        self.dir_inputs = dir_inputs
        self.dir_labels = dir_labels
        self.channels = 2 if stereo else 1
        self.sample_rate = 48000 if stereo else 24000
        self.max_duration = max_duration

        self.model = tokenizer_model

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx) -> Dict[torch.Tensor]:
        label_path = self.data_path.iloc[idx]["label"]
        midi_path = f"{self.dir_midi}/{label_path}_basic_pitch.mid"
        
        label_wav, label_sr = torchaudio.load(f"{self.dir_labels}/{label_path}")
        
        input_codes = self.tokenizer.encode(midi_path)[0].ids
        
        label_wav = convert_audio(label_wav, label_sr, self.sample_rate, self.channels)
        
        label_wav = self._cut_wave(label_wav, self.max_duration, self.sample_rate)
        
        label_codes = self._get_codes(label_wav)
        
        input_codes = torch.cat([self.tokenizer["BOS_None"], input_codes, self.tokenizer["EOS_None"]], dim=-1)
        
        label_input = torch.cat([torch.empty(label_input.size(0), 1).fill_(self.bos_token).to(label_input.device).type_as(label_input),
                                label_input], dim=-1)
        
        label_codes = torch.cat([label_codes, 
                                 torch.empty(label_codes.size(0), 1).fill_(self.eos_token).to(label_codes.device).type_as(label_codes)], dim=-1)

        return {
            "input_codes": input_codes,
            "label_input": label_input,
            "label_codes": label_codes
        }
    
    def _get_codes(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frames = self.model.encode(wav.unsqueeze(0).to(self.device))
        return torch.cat([encoded[0] for encoded in frames], dim=-1).squeeze(0)
    
    def _cut_wave(self, wav: torch.Tensor, max_len: int, sample_rate: int) -> torch.Tensor:
        if wav.shape[-1] > max_len * sample_rate:
            wav = wav[:,:max_len * sample_rate]
        return wav
    
        

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
    
    dataset = LowDataset(tokenizer_model, **config["LowDataset"])

    dataset[0]
    print("Test passed!")


if __name__ == "__main__":
    test(config)