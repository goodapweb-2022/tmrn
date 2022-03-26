import torch.nn as nn
from torch.optim import Adam,SGD
import torch

class TMRN_trainer():

    def __init__(self,model,optim='Adam',use_mop=True,use_cuda=True,lr=0.0001,weight_decay=1e-4,momentum=0.9):
        self.use_cuda=use_cuda
        self.model = model
        self.use_mop=use_mop
        if optim== 'Adam':
            self.optim = Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        elif optim=='SGD':
            self.optim = SGD(self.model.parameters, lr=lr, momentum=momentum,
                                        weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self,train_data,epoch_times):

        for i in range(epoch_times):
            total=0
            for data,ba_index,ba_adj,ba_label,ba_label_ in train_data:
                if not self.use_mop:
                    ba_label=ba_label_

                if self.use_cuda:
                    data,ba_index,ba_adj,ba_label=data.cuda(),ba_index.cuda(),ba_adj.cuda(),ba_label.cuda()
                output= self.model.forward(data,ba_index,ba_adj,self.use_cuda)
                loss = self.criterion(output,ba_label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                now = loss.item()
                total += now
            print(str(i)+"  "+str(total))



    def test(self,test_data,ev=2776):
        correct=0
        correct_ev=0
        count=0
        count_ev=0
        nll_=0
        for data,ba_index,ba_adj,ba_label,ba_label_ in test_data:
            if not self.use_mop:
                ba_label = ba_label_
            if self.use_cuda:
                data,ba_index,ba_adj,ba_label=data.cuda(),ba_index.cuda(),ba_adj.cuda(),ba_label.cuda()
            output = self.model.forward(data, ba_index, ba_adj,self.use_cuda)
            predict = self.softmax(output)
            _, predict = torch.max(predict, dim=-1)
            correct+=(predict==ba_label).sum().item()
            count += len(ba_index)
            for i,j in enumerate(ba_label_,0):
                if j==ev:
                    count_ev+=1
                    if predict[i]==ba_label[i]:
                        correct_ev+=1
            nl=self.criterion(output,ba_label).item()*len(data)
            nll_+=nl


        print('accuracy:',correct/count)
        print('nll:',nll_/len(test_data))
        print('ev_accuracy:',correct_ev/count_ev)


    def save(self, file_path):
        output_path = file_path
        torch.save(self.model, output_path)
        print("Model_TMRN Saved on:", output_path)

        return output_path


class Road2vec_trainer():

    def __init__(self, model, lr, use_cuda=True,a=0.5):
        self.model = model
        self.use_cuda=use_cuda
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rate = a
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, train_data, epoch):
        for i in range(epoch):
            l = 0
            for x, y, label1, label2 in train_data:
                if self.use_cuda:
                    x,y,label1,label2=x.cuda(),y.cuda(),label1.cuda(),label2.cuda()
                output = self.model(x, y)
                loss1 = self.criterion(output[0], label1)
                loss2 = self.criterion(output[1], label2)
                loss = self.rate * loss1 + (1 - self.rate) * loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                l += loss.item()
            print(i, l)

    def save(self, output_path):
        torch.save(self.model.encoder, output_path)
        print("Road2vec Saved on: " + output_path)
        return output_path
