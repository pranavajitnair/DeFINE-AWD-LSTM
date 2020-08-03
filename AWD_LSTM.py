import torch.nn as nn
import torch

from embeddings import Adaptive_Tied_Input_Softmax,DeFINE


class VariationalDropout(nn.Module):
        def __init__(self):
                super(VariationalDropout,self).__init__()
                
        def forward(self,x,dropout,training=True):
                if not training:
                        return x
                    
                mask=x.data.new(x.shape[0],1,x.shape[2]).bernoulli_(1-dropout)
                mask=mask/(1-dropout)
                mask=mask.expand(-1,x.shape[1],-1)
                
                return x*mask
            

class AWD_LSTM(nn.Module):
        def __init__(self,embed_dim,h_size,layers,
                     vocab_sizes,m,k,N,g_list,n_layers,
                     dropouti,dropouth,dropout,dropout_embed,dropout_wei):
                super(AWD_LSTM,self).__init__()
                
                self.embeddings=Adaptive_Tied_Input_Softmax \
                (layers,embed_dim,vocab_sizes)
                self.define=DeFINE(m,embed_dim,k,N,g_list)
                
                self.lstms=nn.ModuleList()
                self.n_layers=n_layers
                for i in range(n_layers):
                        self.lstms.append(
                                WeightDrop(nn.LSTM(m if i==0 else h_size,
                                               h_size if i!=n_layers-1 else embed_dim,
                                               batch_first=True),
                                           ['weight_hh_l0'],dropout=dropout_wei))
                
                self.drop=VariationalDropout()
                self.dropouti=dropouti
                self.dropout=dropout
                self.dropouth=dropouth
                self.dropout_embed=dropout_embed
                
        def forward(self,input,hidden):
                input_=input
                # self.embeddings.embed_dropout(self.dropout_embed)
                embeddings=self.embeddings.get_embeds(input)
                embeddings=self.define(embeddings)
                embeddings=self.drop(embeddings,self.dropouti)
                
                raw_output=embeddings
                new_hidden,raw_outputs,outputs=[],[],[]
                for i,module in enumerate(self.lstms):
                        raw_output,h=module(raw_output,hidden[i])
                        raw_outputs.append(raw_output)
                        new_hidden.append(h)
                        if i!=self.n_layers-1:
                                raw_output=self.drop(raw_output,self.dropouth)
                                outputs.append(raw_output)
                                
                hidden=new_hidden
                output=self.drop(raw_output,self.dropout)
                outputs.append(output)
                logits=self.embeddings.get_logits(output,input_)
                
                return logits,hidden,raw_outputs,outputs


class WeightDrop(torch.nn.Module):
        def __init__(self,module,weights,dropout=0,variational=False):
                super(WeightDrop, self).__init__()
                self.module=module
                self.weights=weights
                self.dropout=dropout
                self.variational=variational
                self._setup()

        def widget_demagnetizer_y2k_edition(*args, **kwargs):
            
                return

        def _setup(self):
                if issubclass(type(self.module),torch.nn.RNNBase):
                        self.module.flatten_parameters=self.widget_demagnetizer_y2k_edition

                for name_w in self.weights:
                        w=getattr(self.module,name_w)
                        del self.module._parameters[name_w]
                        self.module.register_parameter(name_w +'_raw',nn.Parameter(w.data))

        def _setweights(self):
                for name_w in self.weights:
                        raw_w=getattr(self.module,name_w + '_raw')
                        w=None
                        if self.variational:
                                mask=torch.autograd.Variable(torch.ones(raw_w.size(0),1))
                                if raw_w.is_cuda:
                                        mask=mask.cuda()
                                mask=torch.nn. \
                                functional.dropout(mask,p=self.dropout,training=True)
                                w=mask.expand_as(raw_w)*raw_w
                        else:
                                w=nn.Parameter \
                                (torch.nn.functional. \
                                 dropout(raw_w,p=self.dropout,training=self.training))
                        setattr(self.module,name_w,w)

        def forward(self, *args):
                self._setweights()
                
                return self.module.forward(*args)