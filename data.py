import torch
class batch_generator():
    def __init__(self , data ,split_ratio , context_size,device):
        self.data=data
        self.train_data=torch.tensor(data[:int(len(data)*split_ratio)])
        self.val_data = torch.tensor(data[int(len(data)*split_ratio):])
        self.context_size=context_size
        self.device=device
    def generate(self,batch_size , split  ):
        data_split = self.train_data if split=="train" else self.val_data 
        random_index=torch.randint(len(data_split)-self.context_size,(batch_size,))
        x = torch.stack([data_split[i:i+ self.context_size] for i in random_index])
        y = torch.stack([data_split[i+1:i+self.context_size+1] for i in random_index])
        x, y = x.to(self.device), y.to(self.device)
        return x,y 
        