from pathlib import Path

def get_config():
    config = {
        "lang_src": "en",
        "lang_tgt": "it",
        "seq_len": 350,
        "batch_size": 8,
        "d_model": 512,
        "num_epochs": 20,
        "lr": 10**-4,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }
    return config

def get_weights_file_path(config,epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)