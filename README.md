# DeFINE-AWD-LSTM

Implementation of the paper [DEFINE: DEEP FACTORIZED INPUT TOKEN EMBEDDINGS FOR NEURAL SEQUENCE MODELING](https://openreview.net/pdf?id=rJeXS04FPH) with AWD-LSTM
from the paper [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf) in PyTorch

This implemmentation only supports Language Modeling on Penn Treebank. Fine tuning for AWD-LSTM has not been implemented.

## Training
To train the model run
```
python train.py
```
Optional arguments
```
--lr               Learning rate
--epochs           Number of epochs over the whole dataset
--k                Size of the last layer of the DeFINE embeddings
--embed_dim        Embedding size for adaptive shared inputs
--h_size           hidden size for LSTM
--layers           Number of splits for vocab
--n_layers         Number of LSTM layers
--path_train       path to training file
--path_dev         path to validation file
--path_test        path to testing file
--dropouti         Variational dropout for input to the first LSTM layer
--dropouth         Variational dropout for input to the other LSTM layer
--dropout          Variational dropout to the output of the final LSTM layer
--dropout_embed    Dropout for embeddings
--N                Number of layers in DeFINE
--t1               First partition frequency for vocabulary
--t2               Second partition frequency for vocabulary
--n0               Minimum validation runs for non-monotonic ASGD
--log_interval     Logging interval while training
--batch_size       Batch size for training
--bptt             Mean for sequence length for training
--alpha            Scaling factor for Activation Regularization
--beta             Scaling factor for Temporal Activation Regularization
--dropoutw         DropConnect for hidden to hidden LSTM weights
--dev_batch_size   Match size for development
--m                Final output embedding size for DeFINE embeddings
--clip             Gradient nomr for clipping
```
