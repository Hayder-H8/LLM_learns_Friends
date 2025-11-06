def read_process ( fraction , path="Friends_Transcript\Friends_Transcript.txt" , encoding_type="utf-8"):
        with open(path) as f: 
            corpus=f.read()
        usage=int(fraction*len(corpus))
        corpus_idx =  list(corpus[:usage].encode(encoding_type))
        return  corpus_idx , max(corpus_idx)  

def count_pairs(corpus_idx):
    dict_pair={}
    for i in range(len(corpus_idx)-1):
        pair=(corpus_idx[i] , corpus_idx[i+1])
        dict_pair[pair]= dict_pair.get(pair,0)+1
    return dict_pair
def rectify(corpus_idx , new_merge , new_token_id):
    i=0
    while i < len(corpus_idx)-1:
        pair=[corpus_idx[i],corpus_idx[i+1]]
        
        if pair==list(new_merge):
            corpus_idx[i]=new_token_id
            corpus_idx.pop(i+1)
        i+=1
    
def merge(corpus_idx, num_merges):
    max_byte=max(corpus_idx)
    print(max_byte)
    merges={}
    for it  in  range(1,num_merges):
        pair_count=count_pairs(corpus_idx)
        sorted_pairs=sorted(((v,k) for k,v in pair_count.items()),reverse=True)
        new_merge=sorted_pairs[0][1]
        
        merges[new_merge] = max_byte+it
        print(f'{new_merge} has been added as token number : {max_byte+it}')
        rectify(corpus_idx , new_merge , max_byte+it)

    return merges

def build_vocab(merges , max_corpus_idx):
        vocab={id:bytes([id]) for id in range(max_corpus_idx+1)}
        for (p1,p2) , idx in merges.items():
            vocab[idx]=vocab[p1]+vocab[p2]
        return vocab
