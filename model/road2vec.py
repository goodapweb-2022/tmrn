import torch
import torch.nn.functional as f

class Road2vec(torch.nn.Module):

    def __init__(self,input_size,embedding_size):
        super(Road2vec, self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.l=torch.nn.Linear(self.input_size,self.embedding_size)

    def forward(self,x):
        x=f.one_hot(x,self.input_size)
        x=x.float()
        x=self.l(x)
        x=f.normalize(x,p=2,dim=-1)
        return x

class Decoder(torch.nn.Module):

    def __init__(self,hidden_size):
        super(Decoder,self).__init__()
        self.l1=torch.nn.Linear(hidden_size,1)
        self.l2=torch.nn.Linear(hidden_size,1)
        self.activaton=torch.nn.Sigmoid()

    def forward(self,x,y):
        output=x.mul(y)
        output1=self.l1(output)
        output1=self.activaton(output1)
        output2=self.l2(output)
        output2=self.activaton(output2)
        return output1,output2

class TrainRoad2vecModel(torch.nn.Module):

    def __init__(self,input_size,embedding_size):
        super(TrainRoad2vecModel,self).__init__()
        self.encoder=Road2vec(input_size,embedding_size)
        self.decoder=Decoder(embedding_size)

    def forward(self,x,y):
        x=self.encoder.forward(x)
        y=self.encoder.forward(y)
        output=self.decoder.forward(x,y)
        return output



