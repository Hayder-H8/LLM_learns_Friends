from tokenizer import byte_pair_tokenizer
from model import model_predictor
from data import batch_generator
from train import train
num_merges=120
context_size=50
#*********************---Training the tokenizer---***********
tokenizer = byte_pair_tokenizer(0.5 ,num_merges)
model=model_predictor(nb_layers=6,vocabulary_length= num_merges+tokenizer.max_byte )
data_idx=tokenizer.encode_using_regex(tokenizer.corpus_idx)
batch_generator=batch_generator(data_idx,0.8 ,50 )

