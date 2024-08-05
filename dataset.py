import pickle
import pandas as pd
import torchaudio
import torch
from typing import Tuple
from config import get_config
from encodec.utils import convert_audio

from torch.utils.data import Dataset


class LowDataset(Dataset):
    def __init__(self, 
                 data_path: pd.DataFrame,
                 tokenizer_model,
                 max_duration: int,
                 max_len_token: int,
                 stereo: bool = False,
                 pad_token: int = 1025,
                 bos_token: int = 1024,
                 eos_token: int = 1026,
                 masking: bool = False,
                 dir_inputs: str = "data/inputs",
                 dir_labels: str = "data/labels",
                 device: str = "cpu"
                 ):
        super().__init__()
        
        with open(data_path, "rb") as f:
            if "jsonl" in data_path:
                self.data_path = pd.read_json(f, orient="records", lines=True)
            elif "csv" in data_path:
                self.data_path = pd.read_csv(f)
            elif "pkl" in data_path:
                self.data_path = pickle.load(f)
            else:
                raise ValueError("Invalid data path")
            
        self.device = device
        
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.masking = masking
        self.dir_inputs = dir_inputs
        self.dir_labels = dir_labels
        self.channels = 2 if stereo else 1
        self.sample_rate = 48000 if stereo else 24000
        self.max_duration = max_duration
        self.max_len_token = max_len_token

        self.model = tokenizer_model

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        input_path = self.data_path.iloc[idx]["input"]
        label_path = self.data_path.iloc[idx]["label"]
        
        input_wav, input_sr = torchaudio.load(f"{self.dir_inputs}/{input_path}")
        label_wav, label_sr = torchaudio.load(f"{self.dir_labels}/{label_path}")
        
        input_wav = convert_audio(input_wav, input_sr, self.sample_rate, self.channels)
        label_wav = convert_audio(label_wav, label_sr, self.sample_rate, self.channels)
        
        input_wav = self._cut_wave(input_wav, self.max_duration, self.sample_rate)
        label_wav = self._cut_wave(label_wav, self.max_duration, self.sample_rate)
        
        input_codes = self._get_codes(input_wav)
        label_codes = self._get_codes(label_wav)
        
        input_codes = torch.cat([torch.empty(input_codes.size(0), 1).fill_(self.bos_token),
                                 input_codes, 
                                 torch.empty(label_codes.size(0), 1).fill_(self.eos_token)], dim=-1)
        
        label_input = torch.cat([torch.empty(label_codes.size(0), 1).fill_(self.bos_token),
                                label_codes], dim=-1)
        
        label_codes = torch.cat([label_codes, 
                                 torch.empty(label_codes.size(0), 1).fill_(self.eos_token)], dim=-1)
        
        input_codes, input_mask = self._padding_codes(input_codes, self.max_len_token, self.pad_token, self.masking)
        label_input, _ = self._padding_codes(label_input, self.max_len_token, self.pad_token, self.masking)
        label_codes, label_mask = self._padding_codes(label_codes, self.max_len_token, self.pad_token, self.masking)

        return {
            "input_codes": input_codes,
            #"input_mask": input_mask,
            "label_input": label_input,
            #"label_mask": label_mask,
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
        
    def _padding_codes(self, codes: torch.Tensor, max_len: int, padding_token : int, masking : bool = False) -> torch.Tensor:
        K, T = codes.shape
        if T < max_len:
            pad = torch.full((K, max_len - T), padding_token).to(codes.device)
            codes = torch.cat([codes, pad], dim=-1)
            mask = (codes == padding_token)[0]
        else: 
            mask = torch.full((T,), False, dtype=torch.bool)
        if masking:
            return codes, mask
        else:
            return codes, None


def test(config):
    from encodec import EncodecModel
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer_model = EncodecModel.encodec_model_24khz().to(DEVICE)
    tokenizer_model.set_target_bandwidth(config["bandwidth"])
    
    dataset = LowDataset(data_path=config["data_path"], 
                        tokenizer_model= tokenizer_model,
                        max_duration=config["max_duration"], 
                        max_len_token=config["max_token_len"],
                        stereo=config["stereo"], 
                        dir_inputs=config["dir_inputs"], 
                        dir_labels=config["dir_labels"], 
                        device=DEVICE)
    data = dataset[0]
    print(data["input_codes"].shape)
    print(data["label_input"].shape)
    print(data["label_codes"].shape)
    print(data["label_mask"].shape)
    print(data["input_mask"].shape)
    print("Test passed!")



if __name__ == "__main__":
    config = get_config()
    test(config)