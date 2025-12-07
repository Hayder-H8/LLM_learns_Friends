import regex as re 
import os
import json
from tokenizer_utils import read_process , count_pairs , merge , rectify , build_vocab

class byte_pair_tokenizer:
    def __init__(self, fraction, num_merges=None,
                 path="Friends_Transcript/Friends_Transcript.txt",
                 tokenizer_path=None):
        self.corpus_idx, self.max_byte = read_process(fraction, path)
        self.splitter = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Case 1: Load from existing JSON file
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"📖 Loading tokenizer data from {tokenizer_path}...")
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert vocab back to bytes (stored as latin1 strings)
            self.vocab = {int(k): v.encode("latin1") for k, v in data["vocab"].items()}

            # Convert merge keys back from strings like "12,45" to tuples (12,45)
            self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}

            print(f"Loaded vocab ({len(self.vocab)}) and merges ({len(self.merges)})")

        # Case 2: Build from scratch
        else:
            if num_merges is None:
                raise ValueError(
                    "You must specify 'num_merges' if no existing tokenizer_path is provided."
                )

            print(f" Building vocabulary from scratch ({num_merges} merges)...")
            self.merges = merge(corpus_idx=self.corpus_idx, num_merges=num_merges)
            self.vocab = build_vocab(self.merges, self.max_byte)

            # Optional save
            if tokenizer_path:
                print(f" Saving vocab and merges to {tokenizer_path}...")
                # Convert bytes → latin1 strings, tuples → string keys
                vocab_json = {k: v.decode("latin1") for k, v in self.vocab.items()}
                merges_json = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

                with open(tokenizer_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"vocab": vocab_json, "merges": merges_json},
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                print(f"Saved tokenizer data to {tokenizer_path}")
                
                
    
    def encode(self , sentence): 
        
        tokens = list(sentence.encode("utf-8"))
        
        merges = self.merges
        
        while True:
            pairs=count_pairs(tokens) 
            if not pairs:
                break
            pair_to_merge= min (pairs ,key=lambda x : merges.get(x , float("inf")) )
            if pair_to_merge not in  merges:
                break
            id=merges[pair_to_merge]
            rectify(tokens , pair_to_merge , id)
        return tokens
    

    def decode(self,token_list ):
        sentence=b"".join(self.vocab[id] for id in token_list )
        sentence=sentence.decode("utf-8" , errors="replace")
        return sentence

    def encode_using_regex(self,sentence):
        tokens=[]
        for i in re.findall(self.splitter,sentence):
            tokens+=self.encode(i)
        return tokens
    
