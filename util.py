import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp
from operator import itemgetter
from collections import Counter
from tqdm import tqdm
import torch

def data_masks(all_sessions, user, n_node, n_user, max_hist_len, train_hist, train_edge_num):
    indptr, indices, data = [], [], []
    u_indptr, u_indices, u_data = [], [], []
    indptr.append(0)
    u_indptr.append(0)
    tmp_hist = []
    cur_user = 0
    global_A = np.diag(np.ones(n_node))
    max_hist = max(Counter(user).values())
    hist_sess = np.array([[0]*max_hist_len for i in range(len(user))])
    hist_len = np.array([0 for i in range(len(user))])
    tmp_len = 0
    max_node = 0
    shift = 0
    if train_hist == None:
        istrain = True
        train_hist = {}
    else:
        shift = train_edge_num
        istrain = False
        tmp_hist = train_hist[user[0]]
        tmp_len = len(tmp_hist)
    for j in range(len(all_sessions)):
        # construct history session
        if cur_user == user[j]:
            if len(tmp_hist) > max_hist_len:
                hist_sess[j] = tmp_hist[-max_hist_len:].copy()#hist_sess[j][:max_hist_len]
                hist_len[j] = max_hist_len
            else:
                hist_sess[j][:tmp_len] = tmp_hist.copy()
                hist_len[j] = tmp_len
            tmp_hist.append(j + shift)
            tmp_len += 1
        else:
            if istrain:
                #train_hist[cur_user] = tmp_hist
                if len(tmp_hist) > max_hist_len:
                    train_hist[cur_user] = tmp_hist[-max_hist_len:].copy()  # hist_sess[j][:max_hist_len]
                else:
                    train_hist[cur_user] = tmp_hist.copy()
                tmp_hist = [j+shift]
                tmp_len = 1
                hist_len[j] = 0
                #hist_sess[j] = tmp_hist
                #tmp_hist.append(j + shift)
            else:
                tmp_hist = train_hist[user[j]]
                tmp_len = len(tmp_hist)
                hist_len[j] = tmp_len
                hist_sess[j][:tmp_len] = tmp_hist
                tmp_hist.append(j + shift)
                tmp_len += 1
            cur_user = user[j]
            #hist_sess[j][:hist_len[j]] = tmp_hist.copy()
        if j == len(all_sessions)-1:
            if len(tmp_hist) > max_hist_len:
                #hist_sess[j] = tmp_hist[-max_hist_len:].copy()#hist_sess[j][:max_hist_len]
                #hist_len[j] = max_hist_len
                train_hist[cur_user] = tmp_hist[-max_hist_len:].copy()
            else:
                #hist_sess[j][:hist_len[j]] = tmp_hist.copy()
                train_hist[cur_user] = tmp_hist.copy()


        # construct global graph
        seq = all_sessions[j]
        if max_node < max(seq):
            max_node = max(seq)
        for i in np.arange(len(seq) - 1):
            u = seq[i]-1
            v = seq[i+1]-1
            global_A[u][v] = 1
            #global_A[v][u] += 1

        # construct hyper graph incidene matrix
        session = np.unique(all_sessions[j]) # 将session转换为集合，损失节点的重复性信息以及访问序关系
        length = len(session)
        s = indptr[-1]
        u_s = u_indptr[-1]
        indptr.append((s + length))
        u_indptr.append((u_s+1))
        u_indices.append(user[j])
        u_data.append(1)
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    u_sum = np.sum(global_A, 0)
    u_sum[np.where(u_sum == 0)] = 1
    global_A = csr_matrix(np.divide(global_A, u_sum[:, None]))

    # indices for positions of items of each session in n_nodes
    # indptr for position slices of each sessions in each row
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    u_matrix = csr_matrix((u_data, u_indices, u_indptr), shape=(len(all_sessions), n_user))

    return matrix, u_matrix, global_A, hist_sess, hist_len, train_hist

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, opt, shuffle=False, n_node=None, n_user=None, train_hist = None, train_edge=None, max_hist_len= 100):
        self.raw = np.asarray(data[0])
        self.user = np.asarray(data[2])
        self.max_hist_len = max_hist_len
        self.opt = opt
        if train_edge != None:
            self.train_edge_num = train_edge.shape[0]
        else:
            self.train_edge_num = 0
        self.train_hist = train_hist
        H_T, u_H_T, global_A, hist_sess, hist_len, self.train_hist = data_masks(self.raw, self.user, n_node, n_user, self.max_hist_len, self.train_hist, self.train_edge_num) # 获取N*E item-hyperedge关联矩阵； M*E user-hyperedge关联矩阵
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1)) # 对每个hyperedge中item进行加权，此处每个item权重相同，为1/hyperedge_degree，此矩阵乘item embedding可得到hyperedge embedding
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)

        u_H = u_H_T.T
        u_UH_T = u_H_T.multiply(1.0/u_H.sum(axis=1).reshape(1, -1)) # 对用户对每条超边加权，此处每个hyperedge权重相同，为1/user_degree， 此矩阵乘hyperedge embedding可得到user embedding
        u_UH = u_UH_T.T
        u_item_matrix = np.dot(u_UH, BH_T)
        # 反向传播
        item_u_matrix = np.dot(DH,u_H_T)

        self.edge_item = BH_T.tocsr()

        if train_edge != None:
            self.edge_item = sp.sparse.vstack((train_edge, self.edge_item))
        if self.opt.ishist:
            hist_item_list = []
            for i in tqdm(range(len(hist_sess))):
                hist_item = self.edge_item[hist_sess[i], :].tocoo()
                u_values = hist_item.data
                u_indices = np.vstack((hist_item.row, hist_item.col))
                i = torch.LongTensor(u_indices)
                v = torch.FloatTensor(u_values)
                shape = hist_item.shape
                hist_item = torch.sparse.FloatTensor(i, v, torch.Size(shape))
                hist_item_list.append(hist_item)
            self.hist_sess_item = np.array(hist_item_list)

        self.adjacency = DHBH_T.tocoo()
        values = self.adjacency.data
        indices = np.vstack((self.adjacency.row, self.adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = self.adjacency.shape
        self.adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.u_adj = u_item_matrix.tocoo()
        self.item_u_adj = item_u_matrix.tocoo()
        u_values = self.u_adj.data
        u_indices = np.vstack((self.u_adj.row, self.u_adj.col))
        i = torch.LongTensor(u_indices)
        v = torch.FloatTensor(u_values)
        shape = self.u_adj.shape
        self.u_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        u_values = self.item_u_adj.data
        u_indices = np.vstack((self.item_u_adj.row, self.item_u_adj.col))
        i = torch.LongTensor(u_indices)
        v = torch.FloatTensor(u_values)
        shape = self.item_u_adj.shape
        self.item_u_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.global_A = global_A.tocoo()
        #self.u_e_mat =
        #self.phase1_mat = BH_T.tocoo()
        #self.phase2_mat = DH.tocoo()


        self.n_node = n_node
        self.n_user = n_user
        self.targets = np.asarray(data[1])
        self.user = np.asarray(data[2])
        self.hist_sess = hist_sess
        self.hist_len = hist_len

        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.user = self.user[shuffled_arg]
            self.hist_sess = self.hist_sess[shuffled_arg]
            self.hist_len = self.hist_len[shuffled_arg]
            if self.opt.ishist:
                self.hist_sess_item = self.hist_sess_item[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node, A, alias_inputs = [], [], [], []
        user = self.user[index]
        inp = self.raw[index]
        batch_hist = self.hist_sess[index]
        batch_hist_len = self.hist_len[index]
        if self.opt.ishist:
            batch_hist_item = self.hist_sess_item[index]
        else:
            batch_hist_item = []
        inputs = []
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node) + 1 # +1 for extra 0
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            node = np.unique([0] + session)
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed([np.where(node == i)[0][0]  for i in session])) + (max_n_node - len(session)) * [0])
            u_A = np.diag(np.ones(max_n_node))#np.zeros((max_n_node, max_n_node))
            # Attentive
            '''
            for i in np.arange(len(u_input) - 1):
                u = np.where(node == u_input[i])[0][0]
                u_A[u][u] = 1
                if u_input[i + 1] == 0:
                    break
                v = np.where(node == u_input[i + 1])[0][0]
                if u == v or u_A[u][v] == 4:
                    continue
                u_A[v][v] = 1
                if u_A[v][u] == 2:
                    u_A[u][v] = 4
                    u_A[v][u] = 4
                else:
                    u_A[u][v] = 2
                    u_A[v][u] = 3
            '''
            # SR-GNN
            for i in np.arange(len(session) - 1):
                #if session[i + 1] == 0:
                #    break
                u = np.where(node == session[i])[0][0]
                v = np.where(node == session[i + 1])[0][0]
                u_A[u][v] = 1
                #u_A[v][u] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()

            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0]  for i in session]+ [0]*(max_n_node-len(session)))

        return self.targets[index]-1, user, session_len, items, reversed_sess_item, mask, A, alias_inputs, batch_hist, batch_hist_len, batch_hist_item
