import argparse
import pickle
import time
from util import Data, split_validation
from new_model import *
from datetime import datetime as dt
from tensorboardX import SummaryWriter
#from preprocess import preprocess_gowalla_lastfm
import os
from preprocess import preprocess_gowalla_lastfm

#D2
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='new_lastfm', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=200, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--max_hist_len', type=int, default=30, help='max history length')
parser.add_argument('--layer', type=int, default=2, help='the number of layer used')
parser.add_argument('--pair_layer', type=float, default=2, help='the number of pair-wise graph layer used')
parser.add_argument('--ishybrid', action="store_true", help='filter incidence matrix')
parser.add_argument('--isnorm', action="store_true", help='normalize embeddings after updating')
parser.add_argument('--ishist', action="store_true", help='historical session')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--device', type=int, default=0, help='gpu device')

#D2
parser.add_argument('--isodps', default=0, type=int, help='the source of data')
parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')

#odps reading

opt = parser.parse_args()
opt.ishist = True
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#torch.cuda.set_device(opt.device)

def main():

    pwd = os.getcwd()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    log_dir = pwd + '/log/' + str(opt.dataset) + '/' + 'basic_' +TIMESTAMP
    writer = SummaryWriter(log_dir)
    n_user = 11479
    opt.isodps = False
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 50000#52727
        n_user = 10000#220000
        #opt.isodps = True
        #opt.layer = 1
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    elif opt.dataset == 'lastfm':
        n_node = 38709
        n_user = 979
        #opt.layer = 1
    elif opt.dataset == 'xing':
        n_node = 59122
        n_user = 11479
        #opt.lr = 0.001
        #opt.embSize = 200
        #opt.layer = 5
    elif opt.dataset == 'reddit':
        n_node = 27453
        n_user = 18271
        #opt.layer = 1
        #opt.embSize = 100
    else:
        n_node = 40000
        n_user = 992

    if opt.isodps == 1:
        import common_io
        #odps reading
        reader = common_io.table.TableReader(
            opt.tables,
            selected_cols="buyer_id,item_id,ds",
            slice_id=0,
            slice_count=1,
            num_threads=1,
            capacity=2048
        )
        total_records_num = reader.get_row_count()#int(reader.get_row_count()/10)  # return 3
        batch_size = 2
        # 读表，返回值将是一个python数组，形式为[(uid, name, price)*2]
        records = reader.read(num_records=total_records_num, allow_smaller_final_batch=False)
        # pd_df = reader.to_pandas()
        print(records[:10])
        matrix = np.array(list(map(np.array, records))).astype('str')
        print(matrix[:10])
        df = pd.DataFrame(matrix, columns=["userId", "itemId", "timestamp"])
        df["timestamp"] = df["timestamp"].astype(int)
        reader.close()

        train_data, test_data = preprocess_gowalla_lastfm(df,1,100000)
        if len(train_data[0]) > 100000:
            n_node = 210000  # 50000#52727
            n_user = 220000

        print("Table preprocess done!")
        # Large Tmall
        '''
        train_data = list(train_data)
        test_data = list(test_data)
        print(len(train_data[0]), len(test_data[0]))
        for d in range(len(train_data)):
            train_data[d] = train_data[d][:int(len(train_data[d])/10)]
        for d in range(len(test_data)):
            test_data[d] = test_data[d][:int(len(test_data[d])/10)]
        print(len(train_data[0]), len(test_data[0]))
        train_data = tuple(train_data)
        test_data = tuple(test_data)
        '''
        #datapath =
    else:
        train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    # train.edge_item: num train session * item num
    train_data = Data(train_data, opt, shuffle=True, n_node=n_node, n_user=n_user, max_hist_len=opt.max_hist_len)
    # train.edge_item: (num train session + num test session) * item num
    test_data = Data(test_data, opt, shuffle=True, n_node=n_node, n_user=n_user, max_hist_len=opt.max_hist_len,
                     train_hist=train_data.train_hist, train_edge=train_data.edge_item)
    print("Data preprocess done!")
    model = trans_to_cuda(DHCN(opt=opt, adjacency=train_data.adjacency,adjacency_2=train_data.adjacency_2, u_adj=train_data.u_adj,
                               global_A=train_data.global_A,
                               n_node=n_node, n_user=n_user, lr=opt.lr, l2=opt.l2, beta=opt.beta,
                               layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, writer, epoch, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            writer.add_scalar('loss/hit%d' % K, metrics['hit%d' % K], epoch)
            writer.add_scalar('loss/mrr%d' % K, metrics['mrr%d' % K], epoch)
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
