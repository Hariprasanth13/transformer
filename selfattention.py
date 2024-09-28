import torch
import torch.nn.functional as F
import math

class SelfAttentionModel():
    def __init__(self) -> None:
        pass

    def self_attention_score(self,h,d_model,seq_len):
        x = torch.rand(1,seq_len,d_model)
        #print(x)
        assert d_model % h ==0, "d_model is not divisible by h"
        
        d_k = d_model//h 
        x = x.view(1,seq_len,h,d_k)   
        x = x.permute(0,2,1,3) # (1,2,4,4)
        print(x)
        q = x
        k=x
        v=x

        attention_scores = torch.matmul(q,k.transpose(-2,-1))
        print(attention_scores)
        scaled_attention_scores = attention_scores/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))

        #applying softmax
        attention_weights = F.softmax(scaled_attention_scores,dim=-1)
        #output = torch.matmul(attention_weights,v.transpose(-2,-1))
        output = attention_weights@ v.transpose(-2,-1)
        output = output.permute(0,2,1,3)
        output =output.reshape(1,seq_len,d_model)
        print(output)

selfattention = SelfAttentionModel()
selfattention.self_attention_score(2,8,4)