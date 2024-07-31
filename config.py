def get_config():
    return {
        # General parameters
        "data_path": "audio_files.jsonl",
        "dir_inputs": "data/inputs",
        "dir_labels": "data/labels",
        "stereo": False,
        "batch_size": 1,
        "masking": False,
        "epochs": 1,
        "lr": 10**-4,
        "max_duration": 90,
        "max_token_len": 7500,
        "codebook_size": 1024 + 2,
        "pad_token": 1025,
        "special_token": 1024,
        "bandwidth": 6.0,
        "codebook_num": 8,
        
        # Transformer parameters
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "d_model": 128,
        "nhead": 4,
        "dropout": 0.1,
        "d_ff": 1024
    }