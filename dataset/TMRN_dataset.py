
import torch
from torch.utils.data import Dataset
import copy
class TMRN_dataset(Dataset):

    def __init__(self,filename,idx_adj,mask_num=2775):
        super(TMRN_dataset,self).__init__()
        with open(filename,'r') as f:
            data = f.read().split("\n")
        self.data=[]
        n=0
        for i in data:
            if i=='':
                continue
            tra = [int(j) for j in i.split('\t') if i != ' ' and i != '']
            self.data.append(tra)
            if n<len(tra):
                n=len(tra)
        self.dic_adj = dict()
        with open(idx_adj, "r") as f:
            a = f.read().split("\n")
            m = 0
            for i in a:
                if i != "":
                    l = i.split("\t")
                    name = l[0]
                    adj = []
                    for j in l[1:]:
                        adj.append(int(j))
                    if len(adj) > m:
                        m = len(adj)
                    self.dic_adj[int(name)] = adj

        self.mask_num=mask_num
        self.tra_max_len=n
        self.adj_max_len=m

    def __getitem__(self, item):
        tra=copy.deepcopy(self.data[item])
        idx=len(tra)-1
        l=tra[-1]
        tra[-1]=self.mask_num
        adj_lst=self.dic_adj[tra[-2]]
        tra=tra+[0]*(self.tra_max_len-len(tra))
        for i,j in enumerate(adj_lst,0):
            if j==l:
                label=i
        adj_lst=adj_lst+[0]*(self.adj_max_len-len(adj_lst))

        return torch.tensor(tra),torch.tensor(idx),torch.tensor(adj_lst),label,l

    def __len__(self):
        return len(self.data)



