import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np
import scipy.sparse as sp
# from misc.utils import positional_encoding


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters() # initialize

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, G, mask=None):
        support = self.fc(input)
        if mask is not None:
            support.masked_fill_(mask.unsqueeze(-1).expand(support.shape), 0)
        output = torch.bmm(G, support)

        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_sim(nn.Module):
    def __init__(self, nfeat, neighbor_k=0):
        super(GCN_sim, self).__init__()
        self.nfeat = nfeat
        self.neighbor_k = neighbor_k
        self.gc1 = GraphConvolution(nfeat, nfeat)
        # self.gc2 = GraphConvolution(nfeat, nfeat)
        # self.gc3 = GraphConvolution(nfeat, nfeat)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU()

        self.fc_layer = nn.Linear(nfeat, nfeat//4)
        self.norm_1 = nn.LayerNorm(nfeat)
        # self.norm_2 = nn.LayerNorm(nfeat)
        # self.norm_3 = nn.LayerNorm(nfeat)

        # self.pos_encoding = positional_encoding(32, nfeat, torch.float).cuda()


    def forward(self, x, mask=None):
        '''
        input X: batch_size, N, d
        mask: batch_size, N
        output X: batch_size, N, d
        '''
        # x = x + self.pos_encoding

        x_norm = self.norm_1(x)
        Gsim = self.cul_Gsim(x_norm, mask)
        x_gc = self.gc1(x_norm, Gsim, mask=mask)

        if mask is not None:
            x_gc.masked_fill_(mask.unsqueeze(-1).expand(x_gc.shape), 0)

        x = x + self.dropout(x_gc)
        return x

    def cul_Gsim(self, X, mask=None):
        X = self.fc_layer(X)
        if mask is not None:
            X.masked_fill_(mask.unsqueeze(-1).expand(X.shape), 0)

        X_norm = F.normalize(X, p=2, dim=-1)
        g_sim = torch.matmul(X_norm, X_norm.transpose(1, 2))  # batch_size, N, N

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # expand (b_z, N) -> (b_z, N, N)
            g_sim = g_sim.masked_fill_(mask, -1e9)

        # if self.neighbor_k > 0:
        #     g_sim = self.process_neighbor(g_sim)
        g_sim = F.softmax(g_sim, dim=-1)
        return g_sim

    def process_neighbor(self, g_sim):
        N = g_sim.size(1)
        k = self.neighbor_k if N>=self.neighbor_k else N

        _, top_k = g_sim.topk(k)
        mask = torch.ones_like(g_sim).byte().cuda()
        for b_s in range(g_sim.size(0)):
            for n in range(N):
                mask[b_s, n][top_k[b_s, n]] = 0
        g_sim[mask] = -1e9
        return g_sim
    
    def __repr__(self):
        return self.__class__.__name__ + ' (N, ' \
               + str(self.nfeat) + ' -> N, ' \
               + str(self.nfeat) + ') neighbor number is ' \
               + str(self.neighbor_k) + ')'
        # return self.__class__.__name__ + ' N, {} -> N, {}, neighbor number is {}.'.format(self.nfeat, self.nfeat, self.neighbor_k)
    
    ##########################################################
    ##########################################################

class GraphConvolution_nobatch(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution_nobatch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters() # initialize

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.weight.data.uniform_(-stdv, stdv)
        self.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, G):
        support = self.fc(input)
        output = torch.mm(G, support)

        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_sim_nobatch(nn.Module):
    def __init__(self, nfeat, neighbor_k=0):
        super(GCN_sim_nobatch, self).__init__()
        self.nfeat = nfeat
        self.neighbor_k = neighbor_k
        self.gc1 = GraphConvolution_nobatch(nfeat, nfeat)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU()

        self.fc_layer = nn.Linear(nfeat, nfeat//4)
        self.norm_1 = nn.LayerNorm(nfeat)

        # self.pos_encoding = positional_encoding(32, nfeat, torch.float).cuda()


    def forward(self, x, test=False):
        '''
        input X: batch_size, N, d
        output X: batch_size, d
        '''
        # x = x + self.pos_encoding

        x_norm = self.norm_1(x)
        Gsim = self.cul_Gsim(x_norm)
        x = x + self.dropout(self.gc1(x_norm, Gsim))

        if test:
            return Gsim
            
        return x

    def cul_Gsim(self, X):
        X = self.fc_layer(X)

        X_norm = F.normalize(X, p=2, dim=-1)
        g_sim = torch.matmul(X_norm, X_norm.t())  # batch_size, N, N

        if self.neighbor_k > 0:
            g_sim = self.process_neighbor(g_sim)
        g_sim = F.softmax(g_sim, dim=-1)
        return g_sim

    def process_neighbor(self, g_sim):
        N = g_sim.size(1)
        k = self.neighbor_k if N>=self.neighbor_k else N

        _, top_k = g_sim.topk(k)
        mask = torch.ones_like(g_sim).byte().cuda()
        for n in range(N):
            mask[n][top_k[n]] = 0
        g_sim[mask] = -1e9
        return g_sim
    
    def __repr__(self):
        return self.__class__.__name__ + ' (N, ' \
               + str(self.nfeat) + ' -> N, ' \
               + str(self.nfeat) + 'neighbor number is ' \
               + str(self.neighbor_k) + ')'

if __name__ == "__main__":
    import torch

    # hyperparameters
    N = 16*10
    d = 2048
    b_z = 32
    class_num = 1000

    # input
    X = torch.zeros(b_z, N, d)  # batch, N, d
    # labels = torch.ones(b_z, dtype=torch.long)

    # model
    model = GCN_sim(d, d)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # GPU
    model.cuda()
    X = X.cuda()

    X = model(X)
    print(X)

    # forward
    # for _ in range(10):
    #     model.train()
    #     optimizer.zero_grad()
    #     # output = model(X, G_front, G_back)
    #     output = model(X)
    #     result = fc(output)

    #     loss_train = F.cross_entropy(result, labels)

    #     loss_train.backward()
    #     optimizer.step()
    #     print('step')
    
    #     print(model.gc1.weight)

    # for n, p in enumerate(model.parameters()):
    #     if p.requires_grad:
    #         print(n)
    #         print(p)
    #         print(p.grad)

