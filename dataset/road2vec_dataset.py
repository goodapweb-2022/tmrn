from torch.utils.data import Dataset

class Road2vec_dataset(Dataset):

    def __init__(self,idx):
        with open(idx) as f:
            a = f.read()
            self.data=a.split('\n')
            if self.data[-1]=='':
                self.data=self.data[0:-1]

    def __getitem__(self, item):

        t=self.data[item].split(',')

        return int(t[0]),int(t[1]),int(t[2]),int(t[3])

    def __len__(self):

        return len(self.data)



