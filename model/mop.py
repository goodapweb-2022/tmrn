import torch
import torch.nn as nn

class Mop(nn.Module):

    def __init__(self,road2vec):
        super(Mop,self).__init__()
        self.road2vec=road2vec

    def forward(self, representation,data_adj):
        mask1 = (data_adj > 0)
        output = representation.unsqueeze(dim=1)
        adj_embedding = self.road2vec(data_adj)
        e = torch.transpose(adj_embedding, 1, 2)
        predict = torch.bmm(output, e)
        predict = predict.squeeze(dim=1)
        predict = predict.masked_fill(mask1 == 0, -1e9)

        return predict

