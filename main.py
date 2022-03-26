from args import get_args
from dataset.TMRN_dataset import TMRN_dataset
from dataset.road2vec_dataset import Road2vec_dataset
from torch.utils.data import DataLoader
from model.TMRN import TMRN
from model.road2vec import TrainRoad2vecModel
from train import Road2vec_trainer,TMRN_trainer
import torch
import os


args = get_args().parse_args()
print(args)

road2vec=None
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)


if args.use_cuda:
    torch.cuda.set_device(args.gpu_id)
if args.use_road2vec and args.train_road2vec:
    print('start training road2vec...')
    train_net=TrainRoad2vecModel(args.state_size,args.embed_size)
    if args.use_cuda:
        train_net=train_net.cuda()
    road2vec_dataset = Road2vec_dataset(args.road2vec_path)
    train_road2vec_dataloder = DataLoader(road2vec_dataset, batch_size=args.road2vec_batch_size,
                                              shuffle=True, num_workers=args.num_workers)


    train_road2vec=Road2vec_trainer(train_net,args.road2vec_lr,a=args.road2vec_rate,use_cuda=args.use_cuda)
    train_road2vec.train(train_road2vec_dataloder,args.road2vec_epoch)
    road2vec=train_net.encoder
    train_road2vec.save(args.output_path+'/road2vec.pth')

elif args.use_road2vec and not args.train_road2vec:
    road2vec=torch.load('./hid288lr0.0002/road2vec.pth')

if args.train:
    train_dataloder=DataLoader(TMRN_dataset(args.data_path,args.adj_path),batch_size=args.batch_size,
                               shuffle=True,num_workers=args.num_workers)
    net = TMRN(args.state_size,args.embed_size,args.hidden_size,use_mop=args.use_mop,
               attention_heads=args.attention_heads,predict=args.predict_model,road2vec=road2vec)

    if args.use_cuda:
        net=net.cuda()

    train_net=TMRN_trainer(net,optim=args.optim,lr=args.learning_rate,use_cuda=args.use_cuda,
                           momentum=args.momentum,weight_decay=args.weight_decay,use_mop=args.use_mop)
    train_net.train(train_dataloder,args.epoch)
    train_net.save(args.output_path+'/TMRN.pth')

if args.test:
    test_dataloder=DataLoader(TMRN_dataset(args.test_path,args.adj_path),batch_size=args.batch_size,
                               shuffle=True,num_workers=args.num_workers)
    net = torch.load(args.output_path + '/TMRN.pth')
    if args.use_cuda:
        net=net.cuda()

    train_net = TMRN_trainer(net, optim=args.optim, lr=args.learning_rate,use_cuda=args.use_cuda,use_mop=args.use_mop)
    train_net.test(test_dataloder)










