import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter, GRU
import torch.nn.functional as F
import torch.sparse

from tqdm import tqdm
import os


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, opt, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.opt = opt
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.GRU = GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        self.h_0 = trans_to_cuda(torch.randn(1, 1, self.emb_size))
        self.W_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, 1))
        self.hist_weight = nn.Parameter(torch.Tensor(30, 1))
        self.seq_range = trans_to_cuda(torch.arange(0, 57).long())
        self.seq_range_expand = self.seq_range.unsqueeze(0).repeat(100, 1)

    def forward(self, adjacency, u_adj, ishist, hist_item, hist_len, embedding, user_embedding, user):

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if self.dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        u_values = u_adj.data
        u_indices = np.vstack((u_adj.row, u_adj.col))
        i = torch.LongTensor(u_indices)
        v = torch.FloatTensor(u_values)
        shape = u_adj.shape
        u_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        item_embeddings = np.sum(final, 0)  # 将n层网络得到的结果取平均作为最后item embeddings

        if self.opt.ishyper:
            user_embeddings = torch.sparse.mm(trans_to_cuda(u_adj), item_embeddings)[user]
            if self.opt.isfixeduser:
                user_embeddings = user_embedding[np.ones_like(user)]
        else:
            if self.opt.isfixeduser:
                user_embeddings = user_embedding[np.ones_like(user)]
            else:
                user_embeddings = user_embedding[user]

        return item_embeddings, user_embeddings


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.issrgnn = True

        # GCN
        self.heads = 1
        self.training = True
        self.dropout = 0.5
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.linear_gcn_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gcn_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lin_att = torch.nn.Linear(self.hidden_size, 1)
        self.a_0 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.lin_W_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.lin_U_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.lin_v_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)

        # Gated
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def Norm(self, A):
        input_in = A[:, :, :A.shape[1]]
        input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]

        input_in = input_in + trans_to_cuda(
            torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))
        input_out = input_out + trans_to_cuda(
            torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))

        deg_in = input_in.sum(2).reshape(input_in.size(0), input_in.size(1), 1).repeat(1, 1, input_in.size(1))
        deg_out = input_out.sum(2).reshape(input_in.size(0), input_in.size(1), 1).repeat(1, 1, input_in.size(1))

        D_in = torch.pow(deg_in, -1)
        D_out = torch.pow(deg_out, -1)


        norm_in = D_in * input_in
        norm_out = D_out * input_out


        return torch.cat([norm_in, norm_out], 2)

    def GCN(self, adj, features, mask):
        shift_mask = torch.cat((trans_to_cuda(torch.zeros(len(mask)).reshape(-1, 1)),
                                mask[:, :adj.shape[1] - 1]), dim=1).float().unsqueeze(-1).repeat(1, 1, adj.shape[1])

        norm = self.Norm(adj)

        hidden_in = torch.matmul(norm[:, :, :norm.shape[1]], features)
        hidden_out = torch.matmul(norm[:, :, norm.shape[1]: 2 * norm.shape[1]], features)

        out = self.linear(torch.cat([hidden_in, hidden_out], 2))

        return out

    def GNNCell(self, A, hidden):

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = hidden + inputgate * (newgate - hidden)
        return hy

    def forward(self, A, hidden, mask):
        if self.issrgnn:
            hidden = self.GNNCell(A, hidden)
        else:
            hidden = self.GCN(A, hidden, mask)
        return hidden


class LineConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len.float())
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn


class H3GNN(Module):
    def __init__(self, opt, adjacency, adjacency_2, u_adj, global_A, n_node, n_user, lr, layers, l2, dataset,
                 emb_size=100, batch_size=100):
        super(H3GNN, self).__init__()
        self.opt = opt
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_user = n_user
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.norm = opt.isnorm
        self.adjacency = adjacency
        self.adjacency_2 = adjacency_2
        self.u_adj = u_adj
        self.global_A = global_A
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.pos_embedding = nn.Embedding(1000, self.emb_size)
        self.length_embedding = nn.Embedding(self.emb_size, 1)
        self.w_len = nn.Linear(self.emb_size, 200, bias=True)
        self.w_user = nn.Parameter(torch.Tensor(self.emb_size))
        self.w_u = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.HyperGraph = HyperConv(self.opt, self.layers, dataset, self.emb_size)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.GCN = GNN(self.emb_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.u_layer = nn.Linear(2 * self.emb_size, self.emb_size)
        self.recent_alpha = 0

        self.W_1 = nn.Linear(2 * self.emb_size, 1)
        self.W_h1 = nn.Linear(2*self.emb_size, self.emb_size)
        # user embedding
        self.v1 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_3 = nn.Linear(self.emb_size, self.emb_size)
        self.w_4 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.w_f = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.w_u = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.vu = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.v2 = nn.Parameter(torch.Tensor(4 * self.emb_size, self.emb_size))
        self.w_sess = nn.Linear(3 * self.emb_size, self.emb_size, bias=True)
        self.v3 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.item_emb_alpha = torch.Tensor(1)
        self.sess_emb_beta = torch.Tensor(1)
        self.user_emb_ratio = 0
        self.gamma = nn.Parameter(torch.Tensor(1))
        self.gamma = nn.Parameter(torch.Tensor(1))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, user, user_embeddings, session_item, session_len, reversed_sess_item,
                          mask, batch_A, alias, batch_hist_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        batch_item_emb = item_embedding[session_item]
        user_emb = user_embeddings  # [user] #

        get = lambda i: batch_item_emb[i][reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        len = seq_h.shape[1]
        seq_l = torch.stack([batch_item_emb[i][reversed_sess_item[i][0]] for i in range(self.batch_size)])
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)

        # current session embedding capturing transition information
        hs = torch.div(torch.sum(seq_h, 1), session_len.float())  # 将session中所有item embedding取平均
        mask = mask.float().unsqueeze(-1)

        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        sess_inten_emb = torch.sum(beta * seq_h, 1)

        # last item session embedding capturing intention information
        coef = torch.sigmoid(self.w_3(seq_h) + self.w_4(seq_l.unsqueeze(1).repeat(1, len, 1)))
        coef = torch.matmul(coef, self.v1)
        coef = coef * mask
        last_inten_emb = (torch.cat([torch.sum(coef * seq_h, 1), seq_l], 1))

        # select, user_emb
        alpha = torch.sigmoid(torch.matmul(torch.cat([sess_inten_emb, last_inten_emb, user_emb], 1), self.v2))
        self.recent_alpha = alpha
        self.sess_emb_beta = alpha

        if self.opt.encoder == 0:
            select = sess_inten_emb * alpha + (1 - alpha) * self.w_f(last_inten_emb) + self.w_user * (user_emb)
        elif self.opt.encoder == 1:
            select = self.w_f(last_inten_emb) + self.w_user * (user_emb)
        elif self.opt.encoder == 2:
            select = sess_inten_emb + self.w_user * (user_emb)

        norm_user = torch.mean(torch.abs(self.w_user * (user_emb)))
        norm_item = torch.mean(torch.abs(sess_inten_emb * alpha + (1 - alpha) * self.w_f(last_inten_emb)))
        self.user_emb_ratio += (norm_user/(norm_user+norm_item)).item()

        if self.norm:
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding again for b
            self.embedding.weight.data = self.embedding.weight.data.div(
                norms.view(-1, 1).expand_as(self.embedding.weight))

        return select

    def forward(self, session_item, session_len, user, reversed_sess_item,
                mask, adj, u_adj, batch_A, batch_hist, batch_hist_len, batch_hist_item, alias, global_A):
        item_embeddings = self.embedding.weight
        if self.opt.ishybrid:
            u_values = global_A.data
            u_indices = np.vstack((global_A.row, global_A.col))
            i = torch.LongTensor(u_indices)
            v = torch.FloatTensor(u_values)
            shape = global_A.shape
            global_A = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            for i in range(self.opt.pair_layer):
                global_item_embeddings = torch.sparse.mm(trans_to_cuda(global_A), item_embeddings)
            if self.norm:
                norms = torch.norm(global_item_embeddings, p=2, dim=1).data  # l2 norm over item embedding
                global_item_embeddings.data = global_item_embeddings.data.div(
                    norms.view(-1, 1).expand_as(global_item_embeddings))

        batch_hist_len = trans_to_cuda(torch.LongTensor(batch_hist_len))
        if self.opt.hybrid != 1:
            item_embeddings_hg, user_emb_hg = self.HyperGraph(adj, u_adj, self.opt.ishist, batch_hist_item, batch_hist_len,
                                                          self.embedding.weight, self.user_embedding.weight,
                                                          user)  # self.embedding.weight, self.user_embedding.weight#
        else:
            item_embeddings_hg, user_emb_hg = self.HyperGraph(adj, u_adj, self.opt.ishist, batch_hist_item, batch_hist_len,
                                                          global_item_embeddings, self.user_embedding.weight,
                                                          user)  # self.embedding.weight, self.user_embedding.weight#
        if self.norm:
            norms = torch.norm(item_embeddings_hg, p=2, dim=1).data  # l2 norm over item embedding
            item_embeddings_hg = item_embeddings_hg.div(
                norms.view(-1, 1).expand_as(item_embeddings_hg))
            norms = torch.norm(user_emb_hg, p=2, dim=1).data  # l2 norm over item embedding
            user_emb_hg = user_emb_hg.div(
                norms.view(-1, 1).expand_as(user_emb_hg))
        if (self.opt.ishybrid and self.opt.ishyper):
            if self.opt.hybrid == 0: # Attention hybrid
                alpha = torch.sigmoid(self.W_1(torch.cat([global_item_embeddings, item_embeddings_hg], 1)))
                self.item_emb_alpha = alpha
                item_embeddings = alpha * item_embeddings_hg + (1 - alpha) * global_item_embeddings
                if self.norm:
                    norms = torch.norm(item_embeddings, p=2, dim=1).data  # l2 norm over item embedding
                    item_embeddings.data = item_embeddings.data.div(
                        norms.view(-1, 1).expand_as(item_embeddings))
            elif self.opt.hybrid == 1: # Cascade Hybrid
                item_embeddings = item_embeddings_hg
            elif self.opt.hybrid == 2: # MLP Hybrid
                item_embeddings = self.W_h1(torch.cat([global_item_embeddings, item_embeddings_hg],1))
            elif self.opt.hybrid == 3:  # Addition Hybrid
                item_embeddings = global_item_embeddings + item_embeddings_hg
            elif self.opt.hybrid == 4: # Mean Hybrid
                item_embeddings = (global_item_embeddings + item_embeddings_hg)/2
        elif self.opt.ishyper:
            item_embeddings = item_embeddings_hg
        elif self.opt.ishybrid:
            item_embeddings = global_item_embeddings



        sess_emb_hgnn = self.generate_sess_emb(item_embeddings, user, user_emb_hg, session_item, session_len,
                                               reversed_sess_item, mask, batch_A, alias, batch_hist_len)

        return item_embeddings, sess_emb_hgnn


def forward(model, i, data):
    tar, user, session_len, session_item, reversed_sess_item, mask, batch_A, alias, batch_hist, batch_hist_len, batch_hist_item = data.get_slice(
        i)
    adj = model.adjacency_2
    u_adj = model.u_adj
    global_A = model.global_A
    batch_A = trans_to_cuda(torch.Tensor(batch_A).float())
    alias = trans_to_cuda(torch.Tensor(alias).long())
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn = model(session_item, session_len, user,
                                                 reversed_sess_item, mask, adj, u_adj, batch_A, batch_hist,
                                                 batch_hist_len, batch_hist_item,
                                                 alias, global_A)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, batch_hist_len


def train_test(model, epoch, train_data, val_data, test_data, best_perform):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    ind = 0
    updates_per_epoch = int(len(slices))
    for i in tqdm(slices):
        model.zero_grad()
        targets, scores, _ = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss.backward()
        model.optimizer.step()
        ind += 1
        par_5 = int(len(slices)/5)-1
        if ind % par_5 == 0:
            model.eval()
            val_slices = val_data.generate_batch(model.batch_size)
            top_K = [5, 10, 20]
            metrics = {}
            for K in top_K:
                metrics['hit%d' % K] = []
                metrics['mrr%d' % K] = []
            for v_sess in val_slices:
                tar, scores, _ = forward(model, v_sess, val_data)
                val_loss = model.loss_function(scores + 1e-8, targets)
                scores = trans_to_cpu(scores).detach().numpy()
                tar = trans_to_cpu(tar).detach().numpy()
                index = np.argsort(-scores, 1)
                for K in top_K:
                    for prediction, target in zip(index[:, :K], tar):
                        metrics['hit%d' % K].append(np.isin(target, prediction))
                        if len(np.where(prediction == target)[0]) == 0:
                            metrics['mrr%d' % K].append(0)
                        else:
                            metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
            for K in top_K:
                metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
                metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100

            print('\t---After %d steps' % (ind),
                  'train_loss:%.4f\tvalid_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tMMR@5:%.4f'
                  '\tMrr@10:%.4f\tMMR@20:%.4f' % (loss, val_loss,
                metrics['hit5'], metrics['hit10'], metrics['hit20'], metrics['mrr5'], metrics['mrr10'], metrics['mrr20']))
            if (metrics['hit10']+metrics['mrr10'] > best_perform):
                best_perform = metrics['hit10']+metrics['mrr10']
                if os.path.exists("./ckpt"):
                    torch.save(model, "./ckpt/" + model.opt.dataset+"_ckpt.pt")
                else:
                    os.makedirs("./ckpt")
                    torch.save(model, "./ckpt/" + model.opt.dataset + "_ckpt.pt")
        #total_loss += loss
        del loss
    model.user_emb_ratio /= updates_per_epoch
    print(torch.mean(model.item_emb_alpha))
    print(torch.mean(model.sess_emb_beta))
    print(model.user_emb_ratio)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    hist_len_acc = {}
    hist_len_count = [0] * 500
    best_model = torch.load("./ckpt/"+model.opt.dataset+"_ckpt.pt")
    best_model.eval()
    for i in tqdm(slices):
        tar, scores, batch_hist_len = forward(best_model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        j = 0
        k = 10
        unit = 50
        for prediction, target in zip(index[:, :k], tar):

            if int(batch_hist_len[j]/unit) in hist_len_acc:
                hist_len_acc[int(batch_hist_len[j]/unit)].append(np.isin(target, prediction))
            else:
                hist_len_acc[int(batch_hist_len[j]/unit)] = [np.isin(target, prediction)]
            j += 1
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    for keys in hist_len_acc:
        hist_len_count[keys] = len(hist_len_acc[keys])
        hist_len_acc[keys] = np.mean(hist_len_acc[keys]) * 100
    for i in sorted(hist_len_acc):
        print(str(i) + '\t' + str(hist_len_acc[i]) + '\t' + str(hist_len_count[i]) + '\t' + str(
            hist_len_count[i] / np.sum(hist_len_count[i])))
    return metrics, total_loss, best_perform


