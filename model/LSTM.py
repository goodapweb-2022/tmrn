import torch
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(torch.nn.Module):

    def __init__(self,embed_size,hidden_size,n_layers):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.lstm=torch.nn.LSTM(embed_size,hidden_size,n_layers,batch_first=True)
        self.line=torch.nn.Linear(hidden_size,embed_size)
        self.tanh=torch.nn.Tanh()

    def start_hidden_c(self,batch_size):
        hidden=torch.zeros(self.n_layers,batch_size,self.hidden_size).cuda()
        c=torch.zeros(self.n_layers,batch_size,self.hidden_size).cuda()
        return hidden,c

    def forward(self,input_data,data_index):
        batch_size=input_data.size(0)
        start_h,start_c=self.start_hidden_c(batch_size)
        output,_=self.lstm(input_data,(start_h,start_c))
        output = output[list(range(batch_size)),data_index]
        output=self.tanh(output)
        output=self.line(output)
        return output




