import torch.nn as nn
from model.LWA import LWA
from model.mop import Mop
from model.LSTM import LSTM
class TMRN(nn.Module):

    def __init__(self,state_size,embed_size,hidden_size,predict='LWA',road2vec=None,
                 use_mop=True,attention_heads=3,n_layers=1,drop_out=0.1):
        super(TMRN,self).__init__()
        if road2vec == None:
            self.embedding=nn.Embedding(state_size,embed_size)
        else:
            self.embedding=road2vec
        if predict == 'LWA':
            self.predict_model= LWA(embed_size,hidden_size,attention_heads,drop_out)
        elif predict == 'LSTM':
            self.predict_model = LSTM(embed_size,hidden_size,n_layers)
        if use_mop == True:
            self.mop = Mop(self.embedding)
        else:
            self.encoder = nn.Sequential(nn.Linear(embed_size,hidden_size),nn.Tanh(),nn.Linear(hidden_size,state_size))
        self.predict = predict
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_mop = use_mop

    def forward(self, x,data_index,data_adj,use_cuda=True):

        if self.predict=='LWA':
            y=self.embedding(x)
            representation=self.predict_model(x,y,data_index,use_cuda)
        elif self.predict=='LSTM':
            y=self.embedding(x)
            representation=self.predict_model(y,data_index)
        if self.use_mop:
            predict=self.mop(representation,data_adj)
        else:
            predict=self.encoder(representation)

        return predict


