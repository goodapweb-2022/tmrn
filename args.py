import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = './data1/out_train.txt', type = str)
    parser.add_argument('--adj_path',default='./data1/adj.txt',type=str)
    parser.add_argument('--epoch', default = 32, type = int)
    parser.add_argument('--batch_size', default = 64, type = int)
    parser.add_argument('--optim', default = 'Adam', type = str)
    parser.add_argument('--learning_rate', default = 0.0002, type = float)
    parser.add_argument('--weight_decay', default = 1e-4, type = float)
    parser.add_argument('--drop_out',default=0.06,type=float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--use_cuda', default = True, type = bool)
    parser.add_argument('--gpu_id',default=1,type=int)
    parser.add_argument('--num_workers',default=8,type=int)
    parser.add_argument('--state_size',default=2778,type=int)
    parser.add_argument('--embed_size',default=54,type=int)
    parser.add_argument('--hidden_size',default=288,type=int)
    parser.add_argument('--attention_heads',default=9,type=int)
    parser.add_argument('--use_road2vec',default=True,type=bool)
    parser.add_argument('--train_road2vec',default=False,type=bool)
    parser.add_argument('--road2vec_path',default='./data1/road2vec.txt',type=str)
    parser.add_argument('--road2vec_lr',default=0.0001,type=float)
    parser.add_argument('--road2vec_batch_size',default=32,type=int)
    parser.add_argument('--road2vec_rate',default=0.5,type=float)
    parser.add_argument('--road2vec_epoch',default=100,type=int)
    parser.add_argument('--predict_model',default='LWA',type=str)
    parser.add_argument('--use_mop',default=True,type=bool)
    parser.add_argument('--test',default=True,type=bool)
    parser.add_argument('--test_path',default='./data1/out_test.txt',type=str)

    parser.add_argument('--train',default=True,type=bool)
    parser.add_argument('--output_path',default='./head9',type=str)


    return parser
