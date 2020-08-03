import torch
import torch.nn as nn


class Adaptive_Tied_Input_Softmax(nn.Module):
        def __init__(self,layers,dim,vocab_sizes):
                super(Adaptive_Tied_Input_Softmax,self).__init__()
                self.layers=layers
                self.dim=dim
                self.vocab_sizes=vocab_sizes
                
                self.embeds=nn.ModuleList()
                self.projcs=nn.ModuleList()
                for layer in range(layers):
                        vocab_dim=self.dim//(4**layer)
                        self.embeds.append(nn.Embedding(vocab_sizes[layer],vocab_dim))
                        self.projcs.append(nn.Linear(vocab_dim,self.dim,bias=False))
                        
                self.classes=nn.Linear(self.dim,self.layers-1)
                
        def embed_dropout(self,p):
                for embed in self.embeds:
                        mask=embed.weight.data.new(). \
                        resize_((embed.weight.shape[0],1)). \
                        bernoulli_(1-p).expand_as(embed.weight)/(1-p)
                        masked_embed_weight=mask*embed.weight
                        embed.weight=nn.Parameter(masked_embed_weight)
                        
                        padding_idx=embed.padding_idx
                        if padding_idx is None:
                                embed.padding_idx=-1
        
        def get_embeds(self,input):
                l=[]
                vocab_sizes=[]
                vocab_sizes.append(self.vocab_sizes[0])
                for i in range(1,len(self.vocab_sizes)):
                        vocab_sizes.append(vocab_sizes[i-1]+self.vocab_sizes[i])
                        
                for i in range(input.shape[0]):
                    l1=[]
                    for j in range(input.shape[1]):
                            ind=input[i][j]
                            fl,val=0,0
                            for layer in range(self.layers):
                                    if ind<vocab_sizes[layer] and layer==0:
                                            val,fl=ind,0
                                    elif ind>=vocab_sizes[layer-1] and ind<vocab_sizes[layer]:
                                            val,fl=ind-vocab_sizes[layer-1],layer
                                            
                            embed=self.embeds[fl](val.view(1))
                            embed=self.projcs[fl](embed)
                            l1.append(embed)
                    embedi=torch.cat(l1,dim=0).unsqueeze(0)
                    l.append(embedi)
                    
                embeddings=torch.cat(l,dim=0)
                            
                return embeddings
            
        def get_logits(self,hidden,input):
                head_weight=self.embeds[0].weight.unsqueeze(0). \
                expand(input.shape[0],-1,-1).transpose(-2,-1)
                logits_head=torch.bmm(hidden,head_weight)
                l_tail=self.classes(hidden)
                l=[]
                l.append(logits_head)
                
                for layer in range(1,self.layers):
                        embed_weight=self.embeds[layer].weight.unsqueeze(0). \
                        expand(input.shape[0],-1,-1).transpose(-2,-1)
                        proj_weight=self.projcs[layer].weight.unsqueeze(0). \
                        expand(input.shape[0],-1,-1)
                        
                        tail_i=torch.bmm(hidden,proj_weight)
                        tail_i=torch.bmm(tail_i,embed_weight)
                        add_=l_tail[:,:,layer-1].view(hidden.shape[0],hidden.shape[1],1)
                        tail_i+=add_
                        l.append(tail_i)
                        
                logits=torch.cat(l,dim=-1)
                
                return logits
            
            
class DeFINE(nn.Module):
        def __init__(self,m,n,k,N,g_list):
                super(DeFINE,self).__init__()
                self.g_list=g_list
                self.weights=nn.ModuleList()
                mapping=(k-n)//(N-1)+1
                for i,g in enumerate(g_list):
                        self.weights. \
                        append(nn.Linear((n+i*mapping+n*(i!=0))//g,(n+(i+1)*mapping)//g))
                        
                self.finalProj=nn.Linear(n+(N-1)*mapping,m)
                        
        def forward(self,embeddings):
                original=embeddings
                for i,module in enumerate(self.weights):
                        embeddings=embeddings. \
                        view(embeddings.shape[0],embeddings.shape[1],
                             self.g_list[i],embeddings.shape[2]//(self.g_list[i]))
                        
                        if i!=0:
                                original_=original.view \
                                (original.shape[0],original.shape[1], \
                                 self.g_list[i],original.shape[2]//(self.g_list[i]))
                                
                                embeddings=torch.cat([embeddings,original_],dim=-1)
                        embeddings=module(embeddings)
                        embeddings=embeddings.view(embeddings.shape[0],embeddings.shape[1],-1)
   
                embeddings=self.finalProj(embeddings)
                        
                return embeddings