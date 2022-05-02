import datetime
import math
import numpy as np
from scipy import sparse
import torch
from torch import nn, backends
from torch.nn import Module, Parameter, GRU
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from tqdm import tqdm
from collections import Counter

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
    def __init__(self, layers,dataset,emb_size=100, batch_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.dataset = dataset
        self.GRU = GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        self.h_0 = trans_to_cuda(torch.randn(1, 1, self.emb_size))
        self.W_1 = nn.Parameter(torch.Tensor(2*self.emb_size,1))
        self.W_2 = nn.Linear(2*self.emb_size, 200)
        self.seq_range = trans_to_cuda(torch.arange(0, 57).long())
        self.seq_range_expand = self.seq_range.unsqueeze(0).repeat(100,1)
        self.gamma_k = nn.Parameter(torch.Tensor(layers+1))


    def forward(self, adjacency, u_adj, item_u_adj, ishist, hist_item, hist_len, embedding, user_embedding, user):
        '''
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
        '''
        #adjacency = torch.FloatTensor(adjacency)
        item_embeddings = embedding
        final = [item_embeddings*self.gamma_k[0]]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)*self.gamma_k[i+1]
            final.append(item_embeddings)
            #if self.norm:
        item_embeddings = np.sum(final, 0) #将n层网络得到的结果取平均作为最后item embeddings
        '''
        norms = torch.norm(item_embeddings, p=2, dim=1).data  # l2 norm over item embedding again for b
        item_embeddings.data = item_embeddings.data.div(
            norms.view(-1, 1).expand_as(item_embeddings))
        '''

        user_embeddings = user_embedding[user]#torch.sparse.mm(trans_to_cuda(u_adj), item_embeddings)[user]
        if ishist:
            #hist_emb = []
            #for i in range(len(hist_item)):
            '''
            hist = hist_item[i]
            u_values = hist.data
            u_indices = np.vstack((hist.row, hist.col))
            i = torch.LongTensor(u_indices)
            v = torch.FloatTensor(u_values)
            shape = hist.shape
            hist = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            '''
            hist_emb = torch.sparse.mm(trans_to_cuda(torch.cat(tuple(hist_item))), item_embeddings).reshape(self.batch_size,-1,item_embeddings.shape[1])
            #hist_emb.append(hist_embeddings)
            #hist_emb = torch.stack(hist_emb)

            seq_len_expand = trans_to_cuda(hist_len.unsqueeze(1).repeat(1, hist_emb.shape[1]))
            mask = (self.seq_range_expand[:len(seq_len_expand),:hist_emb.shape[1]] >= seq_len_expand)
            neg_value = trans_to_cuda(torch.ones(mask.shape)) * (mask) * -99999
            #h_0 = trans_to_cuda(torch.randn(1, hist_emb.shape[0], self.emb_size))
            #output, h1 = self.GRU(hist_emb, h_0)
            #output = torch.cat((self.h_0.repeat(hist_emb.shape[0],1,1), output), 1)
            #hist_rnn_emb = torch.diagonal(output[:,hist_len],0,0,1).t()
            #hist_rnn_emb#
            query_user = user_embeddings.unsqueeze(1).repeat(1, hist_emb.shape[1], 1)
            alpha = torch.matmul(torch.cat((query_user,hist_emb),2), self.W_1).squeeze(2)+neg_value
            alpha = torch.softmax(alpha, 1).unsqueeze(2)
            hist_len[hist_len==0] = 1#hist_emb.shape[1]
            user_embeddings = torch.sum(alpha*hist_emb, 1)/hist_len.unsqueeze(1) # user_embeddings
        else:
            user_embeddings = torch.sparse.mm(trans_to_cuda(u_adj), item_embeddings)
            new_item_embeddings = torch.sparse.mm(trans_to_cuda(item_u_adj), user_embeddings)
            user_embeddings = user_embeddings[user]
            #alpha = torch.sigmoid(self.W_2(torch.cat([item_embeddings, new_item_embeddings], 1)))
            #item_embeddings = alpha * new_item_embeddings + (1 - alpha) * item_embeddings
            item_embeddings = 0.99*new_item_embeddings+0.01*item_embeddings#(item_embeddings + new_item_embeddings)/2

        return item_embeddings, user_embeddings

class GNN(Module):
    def __init__(self, hidden_size, step=1, issrgnn = False):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.issrgnn = True

        #GCN
        self.heads = 1
        self.training = True
        self.dropout = 0.5
        self.linear = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        self.linear_gcn_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gcn_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lin_att = torch.nn.Linear(self.hidden_size,1)
        #self.leakey_relu = torch.nn.functional.leaky_relu(self.hidden_size)
        self.a_0 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.lin_W_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.lin_U_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.lin_v_out = torch.nn.Linear(self.hidden_size, 1 * self.hidden_size, bias=False)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)

        #Gated
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
        #input_io = A[:, :, 2*A.shape[1]: 3 * A.shape[1]]
        #input_self = A[:, :, 3 * A.shape[1]: 4 * A.shape[1]]

        input_in = input_in + trans_to_cuda(
            torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))
        input_out = input_out + trans_to_cuda(
            torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))
        #input_io = input_io + trans_to_cuda(
        #    torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))
        #input_self = input_self + trans_to_cuda(
        #    torch.eye(input_in.size(1)).reshape(1, input_in.size(1), input_in.size(1)).repeat(input_in.size(0), 1, 1))

        deg_in = input_in.sum(2).reshape(input_in.size(0),input_in.size(1),1).repeat(1,1,input_in.size(1))
        deg_out = input_out.sum(2).reshape(input_in.size(0),input_in.size(1),1).repeat(1,1,input_in.size(1))
        #deg_io = input_io.sum(2).reshape(input_in.size(0), input_in.size(1), 1).repeat(1, 1, input_in.size(1))
        #deg_self = input_self.sum(2).reshape(input_in.size(0), input_in.size(1), 1).repeat(1, 1, input_in.size(1))

        D_in = torch.pow(deg_in, -1)
        D_out = torch.pow(deg_out, -1)
        #D_io = torch.pow(deg_io, -1)
        #D_self = torch.pow(deg_self, -1)

        norm_in = D_in*input_in
        norm_out = D_out*input_out
        #norm_io = D_io * input_io
        #norm_self = D_self * input_self

        return torch.cat([norm_in, norm_out],2) #, norm_io, norm_self

    def GCN(self, adj, features, mask):
        shift_mask = torch.cat((trans_to_cuda(torch.zeros(len(mask)).reshape(-1,1)),
                                mask[:,:adj.shape[1]-1]),dim=1).float().unsqueeze(-1).repeat(1,1,adj.shape[1])

        norm = self.Norm(adj)

        hidden_in = torch.matmul(norm[:, :, :norm.shape[1]], features)
        hidden_out = torch.matmul(norm[:, :, norm.shape[1]: 2 * norm.shape[1]], features)

        out = self.linear(torch.cat([hidden_in, hidden_out],2)) #, hidden_io, hidden_self

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
        hy = hidden + inputgate*(newgate-hidden)#newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden, mask):
        #for i in range(self.step):
        if self.issrgnn:
            hidden = self.GNNCell(A, hidden)#self.GCN(A, hidden, mask)
        else:
            hidden = self.GCN(A, hidden, mask)
        return hidden

class LineConv(Module):
    def __init__(self, layers,batch_size,emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.gamma_k = nn.Parameter(torch.Tensor(layers))
    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1,self.emb_size).fill_(0)
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
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)*self.gamma_k[i]
            session.append(session_emb_lgcn)
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn


class DHCN(Module):
    def __init__(self, opt, adjacency, u_adj, item_u_adj, global_A, n_node, n_user, lr, layers,l2, beta,dataset,emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.opt = opt
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_user = n_user
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.norm = opt.isnorm
        self.adjacency = adjacency
        self.u_adj = u_adj
        self.item_u_adj = item_u_adj
        self.global_A = global_A
        #self.edge_item = edge_item
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, self.batch_size)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.GCN = GNN(self.emb_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.u_layer = nn.Linear(2 * self.emb_size, self.emb_size)

        self.W_1 = nn.Linear(2*self.emb_size, 1)
        # user embedding
        self.v1 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_3 = nn.Linear(self.emb_size, self.emb_size)
        self.w_4 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.w_f = nn.Linear(3*self.emb_size, self.emb_size, bias=False)
        self.w_u = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.vu = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.v2 = nn.Parameter(torch.Tensor(4*self.emb_size, self.emb_size))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

     
    def generate_sess_emb(self, item_embedding, user, user_embeddings, session_item, session_len, reversed_sess_item, mask, batch_A, alias):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        batch_item_emb = item_embedding[session_item]
        #batch_item_emb = self.GCN(batch_A, batch_item_emb, mask)
        user_emb = user_embeddings#[user] #
        get = lambda i: batch_item_emb[i][reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        len = seq_h.shape[1]
        #seq_l = torch.cuda.FloatTensor(self.batch_size, self.emb_size).fill_(0)
        seq_l = torch.stack([batch_item_emb[i][reversed_sess_item[i][0]] for i in range(self.batch_size)])
        #q_u = user_emb.unsqueeze(-2).repeat(1, len, 1)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        #for i in torch.arange(self.batch_size):
        #    seq_l[i] = item_embedding[session_item[i][session_len[i]-1]]
        #current session embedding capturing transition information
        hs = torch.div(torch.sum(seq_h, 1), session_len.float()) # 将session中所有item embedding取平均
        mask = mask.float().unsqueeze(-1)

        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        #length = torch.sum(mask, dim=1)
        #for ind in range(length.size()[0]):
        #    pos_emb[ind][:int(length[ind])] = pos_emb[ind][:int(length[ind])].flip(0)
    
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        sess_inten_emb = torch.sum(beta * seq_h, 1)


        # last item session embedding capturing intention information
        coef = torch.sigmoid(self.w_3(seq_h) + self.w_4(seq_l.unsqueeze(1).repeat(1,len,1)))
        coef = torch.matmul(coef, self.v1)
        coef = coef * mask
        last_inten_emb = (torch.cat([torch.sum(coef * seq_h, 1), seq_l, user_emb], 1))
        #last_inten_emb = self.w_f(torch.cat([torch.sum(seq_h, 1), user_emb], 1))
        '''
        #user session embedding
        gamma = torch.sigmoid(self.w_3(seq_h) + self.w_4(q_u))
        gamma = torch.matmul(gamma, self.v1)
        gamma = gamma * mask
        u_select = torch.sum(gamma * seq_h, 1)
        '''
        # select, user_emb
        alpha = torch.sigmoid(torch.matmul(torch.cat([sess_inten_emb, last_inten_emb], 1), self.v2))
        select = sess_inten_emb*alpha + (1-alpha)*self.w_f(last_inten_emb)

        #beta = torch.sigmoid(torch.matmul(torch.cat([select, ], 1), self.vu))
        #select = select * beta + user_emb * (1-beta)
        #select = self.w_u(torch.cat([select, user_emb], 1))

        #select = self.w_f(torch.cat([sess_inten_emb, torch.sum(coef * seq_h, 1), seq_l, user_emb], 1))
        #u_select = self.u_layer(torch.cat([select, seq_l, user_embedding], -1))
        if self.norm:
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding again for b
            self.embedding.weight.data = self.embedding.weight.data.div(
                norms.view(-1, 1).expand_as(self.embedding.weight))
            #norms = torch.norm(select, p=2, dim=1).data  # l2 norm over item embedding again for b
            #select.data = select.data.div(
            #    norms.view(-1, 1).expand_as(select))

        return select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, session_item, session_len, user, D, A, reversed_sess_item,
                mask, adj, u_adj,item_u_adj, batch_A, batch_hist, batch_hist_len, batch_hist_item, alias, global_A):
        item_embeddings = self.embedding.weight
        if self.opt.ishybrid:
            u_values = global_A.data
            u_indices = np.vstack((global_A.row, global_A.col))
            i = torch.LongTensor(u_indices)
            v = torch.FloatTensor(u_values)
            shape = global_A.shape
            global_A = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            for i in range(self.opt.pair_layer):
                item_embeddings = torch.sparse.mm(trans_to_cuda(global_A), item_embeddings)
            if self.norm:
                norms = torch.norm(item_embeddings, p=2, dim=1).data  # l2 norm over item embedding
                item_embeddings.data = item_embeddings.data.div(
                    norms.view(-1, 1).expand_as(item_embeddings))
        # construct historical edge embedding in batches
        #hist_sess_item = np.array([edge_item[batch_hist[i],:].tocoo() for i in range(100)])
        batch_hist_emb = []
        #batch_hist_len = []

        batch_hist_len = trans_to_cuda(torch.LongTensor(batch_hist_len))

        item_embeddings_hg, user_emb_hg = self.HyperGraph(adj, u_adj, item_u_adj, self.opt.ishist, batch_hist_item, batch_hist_len, self.embedding.weight, self.user_embedding.weight, user)#self.embedding.weight, self.user_embedding.weight#
        if self.norm:
            norms = torch.norm(item_embeddings_hg, p=2, dim=1).data  # l2 norm over item embedding
            item_embeddings_hg = item_embeddings_hg.div(
                norms.view(-1, 1).expand_as(item_embeddings_hg))
            norms = torch.norm(user_emb_hg, p=2, dim=1).data  # l2 norm over item embedding
            user_emb_hg = user_emb_hg.div(
                norms.view(-1, 1).expand_as(user_emb_hg))
        if self.opt.ishybrid:
            alpha = torch.sigmoid(self.W_1(torch.cat([item_embeddings, item_embeddings_hg], 1)))
            item_embeddings = alpha*item_embeddings_hg + (1-alpha)*item_embeddings
            if self.norm:
                norms = torch.norm(item_embeddings, p=2, dim=1).data  # l2 norm over item embedding
                item_embeddings.data = item_embeddings.data.div(
                    norms.view(-1, 1).expand_as(item_embeddings))
        else:
            item_embeddings = item_embeddings_hg

        sess_emb_hgnn = self.generate_sess_emb(item_embeddings, user, user_emb_hg, session_item, session_len, reversed_sess_item, mask, batch_A, alias)
        #session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
        con_loss = 0#self.SSL(sess_emb_hgnn, session_emb_lg)

        return item_embeddings, sess_emb_hgnn, self.beta*con_loss
        #return self.embedding.weight, sess_emb_hgnn, self.beta*con_loss

def forward(model, i, data):
    tar, user, session_len, session_item, reversed_sess_item, mask, batch_A, alias, batch_hist, batch_hist_len, batch_hist_item = data.get_slice(i)
    adj = model.adjacency
    u_adj = model.u_adj
    item_u_adj = model.item_u_adj
    global_A = model.global_A
    #edge_item = model.edge_item
    #batch_adj = trans_to_cuda(torch.Tensor(adj).float())
    #A_hat, D_hat = data.get_overlap(session_item)
    batch_A = trans_to_cuda(torch.Tensor(batch_A).float())
    alias = trans_to_cuda(torch.Tensor(alias).long())
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = 0#trans_to_cuda(torch.Tensor(A_hat))
    D_hat = 0#trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, user, D_hat, A_hat,
                                                 reversed_sess_item, mask, adj, u_adj,item_u_adj, batch_A, batch_hist, batch_hist_len, batch_hist_item,
                                                 alias, global_A)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss, batch_hist_len


def train_test(model, writer, epoch, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    ind = 0
    updates_per_epoch = int(len(slices))
    for i in tqdm(slices):
        model.zero_grad()
        targets, scores, con_loss, _ = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss #+ con_loss
        loss.backward()
#        print(loss.item())
        model.optimizer.step()
        writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + ind)
        ind += 1
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())
    print(model.HyperGraph.gamma_k)
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    len_metric_dict = {}
    len_count = Counter({})
    for i in tqdm(slices):
        tar, scores, con_loss, hist_len = forward(model, i, test_data)
        len_count += Counter(hist_len)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                result = np.isin(target, prediction)
                for i in range(len(hist_len)):
                    length = hist_len[i]
                    if length in len_metric_dict:
                        len_metric_dict[length] += result
                    else:
                        len_metric_dict[length] = np.array(result)
                metrics['hit%d' %K].append(result)
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(prediction == target)[0][0]+1))
    return metrics, total_loss


