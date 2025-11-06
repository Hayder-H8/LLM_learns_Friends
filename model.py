import torch 
import torch.nn as nn
embedding_size=256
context_size=20
dropout=0.3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Head(nn.Module):
    
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(embedding_size,head_size,bias=False)
        self.query = nn.Linear(embedding_size,head_size,bias=False)
        self.value = nn.Linear(embedding_size,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self,x):
        # x of shape (B ,T ,embedding_size)
        B,T,C = x.shape
        key = self.key(x) # (B, T , head_size)
        query = self.query(x)  # (B, T , head_size)
        value = self.value(x)  # (B, T , head_size)
        

        " We don't want past tokens to query future tokens "
        wei = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = nn.Softmax(dim=-1)(wei) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        value = self.value(x) # (B,T,hs)
        out = wei @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class Multiheadattention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    
class Feed_forward_layer(nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(embed_size,embed_size*4),nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout))  
    def forward(self,x):
        return self.network(x)
    
    
class Block(nn.Module):
    def __init__(self, num_heads, embed_size):
        super().__init__() 
        head_size = embed_size//num_heads
        self.Multi_head = Multiheadattention(num_heads , head_size)
        self.thinking_layer=Feed_forward_layer(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    def forward(self,x):
        x=x+self.Multi_head(self.ln1(x))
        x=x+self.thinking_layer(self.ln2(x))
        return x 
        
        

class model_predictor(nn.Module):
    def __init__(self ,nb_layers, vocabulary_length):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocabulary_length,embedding_size)
        self.positional_embedding=nn.Embedding(context_size,embedding_size)
        self.transformer_layer = nn.Sequential(
            *[Block(4, embedding_size) for _ in range(nb_layers)])
        self.final=nn.Linear(embedding_size,vocabulary_length)
        
    def forward(self,idx,target=None):
        T=idx.shape[-1]
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding(torch.arange(T, device=device)) # (T,C)
        
        idx = tok_emb + pos_emb # (B,T,C)
        logits=self.final(self.transformer_layer(idx))
        if target is None:
            loss = None
        else:
            
            logits=logits.view(-1,logits.size(-1))
            target=target.view(-1)
            loss=nn.functional.cross_entropy(logits,target)
    
        return logits , loss 
    def predict(self,idx , num_predictions=50):
        
        for i in range (num_predictions):
            idx_new = idx[:, -context_size:]
            logits , _=self(idx_new)
            logits=logits[:,-1,:]
            probs= nn.Softmax (dim=1)(logits)
            next_index=torch.multinomial(probs , num_samples=1)
            idx=torch.concat((idx,next_index) , dim=1)
        return idx
        