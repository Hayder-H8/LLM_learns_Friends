from tokenizer import byte_pair_tokenizer
from model import model_predictor
from data import batch_generator
from train import train
from tokenizer_utils import read_process
import torch
num_merges=10
context_size=50
nb_layers=6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'every thing will be running on {device}')

#*********************---Training the tokenizer---***********
tokenizer = byte_pair_tokenizer(fraction=0.1 ,num_merges=num_merges , tokenizer_path="vocabulary.json")
vocab_size=len(tokenizer.vocab)

model = model_predictor(nb_layers=nb_layers,vocabulary_length= vocab_size)
model.to(device)
with open("Friends_Transcript\Friends_Transcript.txt") as f: 
            corpus=f.read()

data_idx=tokenizer.encode_using_regex(corpus)

batch_generator=batch_generator(data_idx,split_ratio=0.8 , context_size=20 , device= device)

#*********************---Training the model---***********

train(model=model , batch_generator=batch_generator , iterations=500 , eval_interval=20)


