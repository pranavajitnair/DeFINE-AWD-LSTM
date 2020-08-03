from collections import Counter
import torch

def get_data(path):
        file=open(path)
        data=[]
        
        for lines in file.readlines():
                x=lines.split()
                data.append(x)
        
        return data

def get_vocab(data,t1,t2):
        word_count=Counter()
        for lis in data:
                for word in lis:
                        word_count[word]+=1
                        
        se1,se2,se3=[],[],[]
        for word in word_count:
                if word_count[word]>t1:
                        se1.append(word)
                elif word_count[word]>t2:
                        se2.append(word)
                else:
                        se3.append(word)
                        
        se=[se1,se2,se3]
        word_to_int,int_to_word={},{}
        index=0
        for sei in se:
                for word in sei:
                        word_to_int[word]=index
                        int_to_word[index]=word
                        index+=1
                
        return word_to_int,int_to_word, \
        len(se1),len(se2),len(se3)
    
def tokenize(data,word_to_int):
    
         for i,lis in enumerate(data):
                for j,word in enumerate(lis):
                        if word not in word_to_int: 
                                data[i][j]=word_to_int['<unk>']
                        else:
                                data[i][j]=word_to_int[word]                    
         data=[ind for lis in data for ind in lis]
    
         return data
    
def batchify(data,batch_size):
        temp_len=len(data)//batch_size
        data=data[:batch_size*temp_len]
        data=torch.tensor(data).view(batch_size,-1)
        data=data.t()
        
        return data
    
def get_batch(data,start_index,seq_len):
        seq_len=min(seq_len,data.shape[0]-2-start_index)
        X=data[start_index:start_index+seq_len]
        Y=data[start_index+1:start_index+1+seq_len]
        X,Y=X.t(),Y.t()
        
        return X,Y
    
def repackage_hidden(hidden):
        new_hidden=[]
        for hid in hidden:
                h,c=hid[0].detach(),hid[1].detach()
                new_hidden.append((h,c))
                
        return new_hidden