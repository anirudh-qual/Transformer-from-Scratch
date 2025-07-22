import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split

from dataset import BilingualDataset, casual_mask

from tranformer_model import build_transformer

from config import get_config, get_weights_file_path


from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

    #yield is used to return each sentence one by one from the dataset
    #yield makes a function a generator, which can be iterated over
def get_or_build_tokenizer(config,ds,lang):
    #.format is used to format the path with the language, for example for tokenizer_file input like file_{lang}.json
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Create a trainer for the tokenizer
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # Train the tokenizer on the dataset
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)

        # Save the tokenizer to the specified path
        #str(tokenizer_path) is used to convert the Path object to a string for saving
        tokenizer.save(str(tokenizer_path))
    else:
        # Load the tokenizer from the file if it already exists
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):

    ds_raw = load_dataset("opus_books",f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    #f - tells python to format the string with the values of lang_src and lang_tgt

    #Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source sentences: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
    )
    return model

def train_model(config):
    #Define the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Create weights folder if it doesn't exist
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    #parents=True allows creation of parent directories if they don't exist, exist_ok=True prevents error if the directory already exists

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    #.get_vocab_size() returns the size of the vocabulary of the tokenizer

    #Tensorboard setup
    writer= SummaryWriter(config['experiment_name'])
    #SummaryWriter is used to write logs for TensorBoard, these logs can be visualized in TensorBoard used for monitoring training progress

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],eps=1e-9)

    init_epoch = 0
    global_step = 0

    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state['epoch']+1
        global_step = state['global_step']
        #Global step is used to keep track of the number of training steps
        optimizer.load_state_dict(state['optimizer'])
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    #label_smoothing is used to prevent overfitting -> make model less confident about its predictions

    for epoch in range(init_epoch, config['num_epochs']):
        model.train()
        
