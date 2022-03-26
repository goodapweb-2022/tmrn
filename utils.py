import datetime,os
import random
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir


def get_road2vec_train_data(idx_adj,idx_tra,idx_train_data,state_size,step):
    dic_adj = dict()
    with open(idx_adj,"r") as f1:
        a=f1.read().split("\n")
        for i in a:
            if i!="":
                l = i.split("\t")
                name = l[0]
                ad = l[1:]
                dic_adj[name] = ad
    dic_tra=dict()
    with open(idx_tra ,"r") as f:
        data = f.read().split("\n")
        for j in data:
            tra = j.split(",")
            n = len(tra) - step + 1
            if n > 0:
                for m in range(n):
                    if dic_tra.get(tra[m]) == None:
                        lst = []
                    else:
                        lst = dic_tra[tra[m]]
                    for b in range(1, step):
                        lst.append(tra[m + b])
                    dic_tra[tra[m]]= list(set(lst))

    print(len(dic_tra))
    with open(idx_train_data,"w") as f:
        for name in range(1,state_size):
            val=[]
            name=str(name)
            if dic_tra.get(name)!=None:
                val=dic_tra[name]
            lst = dic_adj[name]
            for j in lst:
                if j not in val:
                    s=name+","+j+","+"0"+","+"1"+"\n"
                    f.write(s)
            for i in val:
                if i in lst:
                    s=name+","+i+","+"1"+","+"1"+"\n"
                else:
                    s=name+","+i+","+"1"+","+"0"+"\n"
                f.write(s)
                p=random.random()
                if p<0.8:
                    a=random.randint(0,state_size)
                    a=str(a)
                    if a not in lst and a not in val:
                        s = name + "," + a + "," + "0" + "," + "0" + "\n"
                        f.write(s)

def get_name_dic(idx,out_idx):
    dic_name=dict()
    count=1
    with open(idx,"r") as f:
        a=f.read().split("\n")
        for i in a:
            if i!="":
                l=i.split(",")
                for name in l :
                    if dic_name.get(name)==None and name!="":
                        dic_name[name]=count
                        count+=1
    print(count-1)
    with open(out_idx,"w") as f1:
        for (key,val) in dic_name.items():
            s=key+"\t"+str(val)+"\n"
            f1.write(s)



def get_new_adj_dic(idx_dic,idx,out_idx):
    dic_name = dict()
    with open(idx_dic, "r") as f:
        t = f.read().split("\n")
        for i in t:
            if i != "":
                i = i.split("\t")
                dic_name[i[0]] = i[1]
    with open(idx, "r") as f:
        t = f.read().split("\n")
        for i in t:
            if i != "":
                i = i.split("\t")
                lst=[]
                if dic_name.get(i[0])!=None:
                    lst.append(dic_name[i[0]])
                    for j in i[1:]:
                        if dic_name.get(j)!=None:
                            lst.append(dic_name[j])
                    lst.append("20227")
                    with open(out_idx,"a") as f1:
                        s="\t".join(lst)+"\n"
                        f1.write(s)






















