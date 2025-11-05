def read_process ( fraction , path="Friends_Transcript\Friends_Transcript.txt" , encoding_type="utf-8"):
    with open("Friends_Transcript\Friends_Transcript.txt") as f: 
        corpus=f.read(path)
    usage=int(fraction*length(corpus))
    return list(corpus[:usage].encode(encoding_type))

def count_pairs(corpus_idx):
    dict_pair={}
    for i in range(len(corpus_idx)-1):
        pair=(corpus_idx[i],corpus_idx[i+1])
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
    merges={}
    for it  in  range(1,num_merges):
        pair_count=count_pairs(corpus_idx)
        sorted_pairs=sorted(((v,k) for k,v in pair_count.items()),reverse=True)
        new_merge=sorted_pairs[0][1]
        merges[new_merge] = max_byte+it
        rectify(corpus_idx , new_merge , max_byte+it)

    return merges


def encode(merges , sentence):
    # Step 1: convert text → bytes
    tokens = list(sentence.encode("utf-8"))
    merges = merge ()
    while True:
        pairs=count_pairs(tokens) 
        if not pairs:
            break
        pair_to_merge= min (pairs ,key=lambda x : merges.get(x , float("inf")) )
        if pair_to_merge not in  merges:
            break
        id=merges[pair_to_merge]
        rectify_list(tokens , pair_to_merge , id)
    return tokens