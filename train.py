import numpy as np 
import torch
from data import batch_generator

def loss_estimate(model,batch_generator,iter_times):
    loss_per_split={}
    for split in ["train","val"]:
        losses=[]
        for i in range(iter_times):
            x,y=batch_generator.generate(split=split,batch_size=1)
            
            _,loss=model(x,y)
            losses.append(loss.item())
        loss_per_split[split]=np.mean(losses)
        
    print(f'The validation loss is {loss_per_split["val"]}')
    
    print(f'The training loss is {loss_per_split["train"]}')
    return loss_per_split["val"]
    
def train(model,batch_generator , iterations=50000 , eval_interval=100):
    optimizer=torch.optim.Adam(model.parameters() , lr=0.001)
    val_losses=[100]
    for epoch in range(iterations):
        optimizer.zero_grad()
        if epoch%eval_interval==0:
            val_loss=loss_estimate(model=model,batch_generator=batch_generator,iter_times=2)
            if val_loss<min(val_losses):
                val_losses.append(val_loss)
                torch.save(model.state_dict(),"llm_friends.pt")
        x,y = batch_generator.generate(split="train",batch_size=32)
        
        logits,loss=model(x,y)
        loss.backward()
        optimizer.step()
    return val_losses