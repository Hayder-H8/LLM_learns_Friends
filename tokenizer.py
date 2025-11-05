import regex as re 
from tokenizer_utils import read_process , count_pairs , merge , rectify , build_vocab
class byte_pair_tokenizer():
    def __init__(self,fraction,path , num_merges ):
        self.corpus_idx , self.max_byte = read_process(fraction,path)
        self.splitter=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        print("f'Creating {num_merges} merges , wait ...")
        self.merges=merge(self.corpus_idx , num_merges)
        self.vocab = build_vocab(self.merges,self.max_byte)

    
    def encode(self , sentence): 
        
        tokens = list(sentence.encode("utf-8"))
        
        merges = self.merges
        
        while True:
            pairs=self.count_pairs(tokens) 
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
    