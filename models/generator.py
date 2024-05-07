import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self , latent_dim = 100 , output_channels_1 = 1024, num_blocks = 4):
        super(Generator, self).__init__()
        
        self.block1 = self.unit_module(2*latent_dim, output_channels_1, stride = 4 , padding= 0)
        self.seq_blocks = nn.Sequential(*[self.unit_module(output_channels_1 // i , output_channels_1 // (2 * i) , 2 , padding = 1) for i in range(1,num_blocks+1) if i%2 == 0 or i == 1])
        self.block2 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride= 2, bias = False , padding = 1)
        self.embedding = nn.Embedding(10, latent_dim)
        self.act = nn.Tanh()
        self.apply(self.__init_weights)
        
    def __init_weights(self, module):
         
        if isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)   
            
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight.data, mean = 0.0, std= 0.02)   
        
         
    def unit_module(self,in_chan,out_chan ,stride, padding):
        
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, kernel_size=4, stride= stride, bias = False , padding = padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        
    def forward(self,x,y):
        
        label_embeddings = self.embedding(y)
        label_embeddings = label_embeddings.view((label_embeddings).shape[0],(label_embeddings).shape[1],1,1)
        x = torch.cat((x,label_embeddings),dim=1)
        x = self.block1(x)
        x = self.seq_blocks(x)
        x = self.block2(x)
        x = self.act(x)
        
        return x
    
