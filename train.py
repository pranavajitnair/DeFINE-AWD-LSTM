import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy.random as random
import os
import numpy as np

from AWD_LSTM import AWD_LSTM
from utils import get_data,get_vocab,tokenize,batchify,
get_batch,repackage_hidden
   
def validate(dev_data,model,seq_len=100):
        model.eval()
        model.drop.training=False
        lossFunction=nn.CrossEntropyLoss()
        iters=dev_data.shape[0]//seq_len
        loss,perplexity=0,0
        hidden=[None for _ in range(model.n_layers)]
          
        for i in range(iters):
            
            if i!=0:  
                    hidden=repackage_hidden(hidden)
         
            X,Y=get_batch(dev_data,i*seq_len,seq_len)
            logits,hidden,raw_outputs,outputs=model(X,hidden)
            logits=logits.view(logits.shape[0]*logits.shape[1],-1)
            Y=Y.contiguous().view(-1)
            loss_=lossFunction(logits,Y)
            loss+=loss_.item()
            perplexity+=np.exp(loss_.item())
        
        return loss/iters,perplexity/iters
            
def train(data_dev,data_train,model,args):
        lossFunction=nn.CrossEntropyLoss()
        optimizer=optim.SGD(model.parameters(),lr=args.lr)
        
        for epoch in range(args.epochs):
                model.train()
                model.drop.training=True
                hidden=[None for _ in range(model.n_layers)]
                i,batch=0,0
                max_loss=1000000000
                best_dev_loss=[]
                
                while i<data_train.shape[0]-2:
                    
                        bptt=args.bptt if random.random()<0.95 else args.bptt//2
                        seq_len=max(5,int(random.normal(bptt,5)))
                        if data_train.shape[0]-2<i+seq_len-1:
                                break
                        lr_=optimizer.param_groups[0]['lr']
                        optimizer.param_groups[0]['lr']=lr_*seq_len/args.bptt
        
                        X,Y=get_batch(data_train,i,seq_len)
                        if i!=0:
                                hidden=repackage_hidden(hidden)
                        optimizer.zero_grad()
                        
                        logits,hidden,raw_outputs,outputs=model(X,hidden)
                        logits=logits.view(logits.shape[0]*logits.shape[1],-1)
                        Y=Y.contiguous().view(-1)
                        loss=lossFunction(logits,Y)
                        ppl_loss=loss.item()
                        loss+=sum(args.alpha*dropped_h.pow(2).mean() \
                                  for dropped_h in outputs[-1])
                        loss+=sum(args.beta*(h[1:]-h[:-1]).pow(2).mean()  \
                                  for h in raw_outputs[-1])
                        scalar=loss.item()
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                        i+=seq_len
                        batch+=1
                        
                        if batch%args.log_interval==0:
                                print('epoch=',epoch+1,'percent done=',i/data_train.shape[0],
                                      'training loss=',scalar,'training perplexity=',np.exp(ppl_loss))
                        
                if 't0' in optimizer.param_groups[0]:
                        temp={}
                        for param in model.parameters():
                                temp[param]=param.data.clone()
                                param.data=optimizer.state[param]['ax'].clone()
                                
                                dev_loss,dev_perplexity=validate(data_dev,model)
                                print('validation loss=',dev_loss,
                                'validation perplexity=',dev_perplexity)
                        
                        if dev_loss<max_loss:
                                max_loss=dev_loss
                                torch.save(model.state_dict(),'model.pth')
                                
                        for param in model.parameters():
                                param.data=temp[param].clone()
                                
                else:
                        dev_loss,dev_perplexity=validate(data_dev,model)
                        print('validation loss=',dev_loss,
                        'validation perplexity=',dev_perplexity)
                
                        if dev_loss<max_loss:
                                max_loss=dev_loss
                                torch.save(model.state_dict(),'model.pth')
                                
                        if 't0' not in optimizer.param_groups[0] and \
                        len(best_dev_loss)>args.n0 and \
                        dev_loss>min(best_dev_loss[:-args.n0]):
                                optimizer=optim.ASGD(modrl.parameters(),t0=0,lr=args.lr)
                                        
def main(args):
        train_data=get_data(args.path_train)
        dev_data=get_data(args.path_dev)
        test_data=get_data(args.path_test)
        
        word_to_int,int_to_word,sz1, \
        sz2,sz3=get_vocab(train_data,args.t1,args.t2)
        train_data=tokenize(train_data,word_to_int)
        dev_data=tokenize(dev_data,word_to_int)
        test_data=tokenize(test_data,word_to_int)
        
        train_data=batchify(train_data,args.batch_size)
        dev_data=batchify(dev_data,args.dev_batch_size)
        test_data=batchify(test_data,args.dev_batch_size)

        train_data=train_data.cuda()
        dev_data=dev_data.cuda()
        test_data=test_data.cuda()
        
        vocab_sizes=[sz1,sz2,sz3]
        g_list=[int(2**(7-i)) for i in range(args.N-1)]
        model=AWD_LSTM(args.embed_dim,args.h_size,args.layers,
                     vocab_sizes,args.m,args.k,args.N,g_list,args.n_layers,
                     args.dropouti,args.dropouth,
                     args.dropout,args.dropout_embed,args.dropoutw).cuda()

        train(dev_data,train_data,model,args)
        test_loss,test_perplexity=validate(test_data,model)
        print('test loss=',test_loss,'test perplexity=',test_perplexity)

def setup():
        parser=argparse.ArgumentParser('argument parser')
        
        parser.add_argument('--lr',type=float,default=30)
        parser.add_argument('--epochs',type=int,default=40)
        parser.add_argument('--k',type=int,default=512)
        parser.add_argument('--embed_dim',type=int,default=128)
        parser.add_argument('--h_size',type=int,default=256)
        parser.add_argument('--layers',type=int,default=3)
        parser.add_argument('--n_layers',type=int,default=3)
        parser.add_argument('--path_train',type=str,default=os.getcwd()+'/data/ptb.train.txt')
        parser.add_argument('--path_dev',type=str,default=os.getcwd()+'/data/ptb.valid.txt')
        parser.add_argument('--path_test',type=str,default=os.getcwd()+'/data/ptb.test.txt')
        parser.add_argument('--dropouti',type=float,default=0.65)
        parser.add_argument('--dropouth',type=float,default=0.3)
        parser.add_argument('--dropout',type=float,default=0.4)
        parser.add_argument('--dropout_embed',type=float,default=0.1)
        parser.add_argument('--N',type=int,default=8)
        parser.add_argument('--t1',type=int,default=1000)
        parser.add_argument('--t2',type=int,default=100)
        parser.add_argument('--n0',type=int,default=5)
        parser.add_argument('--log_interval',type=int,default=10)
        parser.add_argument('--batch_size',type=int,default=50)
        parser.add_argument('--bptt',type=int,default=70)
        parser.add_argument('--alpha',type=float,default=2)
        parser.add_argument('--beta',type=float,default=1)
        parser.add_argument('--dropoutw',type=float,default=0.5)
        parser.add_argument('--dev_batch_size',type=int,default=10)
        parser.add_argument('--m',type=int,default=400)
        parser.add_argument('--clip',type=float,default=0.25)
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)