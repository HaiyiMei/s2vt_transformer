import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np
import scipy.sparse as sp

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, d, init_zero=False):
        super(GraphConvolution, self).__init__()
        # self.N = N
        self.d = d
        self.weight = nn.Parameter(torch.FloatTensor(d, d))  # init 0
        self.bias = nn.Parameter(torch.FloatTensor(d))
        self.reset_parameters() # randomly initialize

    def reset_parameters(self):
        # self.weight.data = torch.eye(self.d)
        # self.bias.data.zero_()
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, G):
        support = torch.mm(input, self.weight)  # N, d
        output = torch.mm(G, support)

        return output + self.bias
    
    def __repr__(self):
        return self.__class__.__name__ + ' N, d -> N, d'

class GCN_sim(nn.Module):
    def __init__(self, d1, d2, neighbor_k=0):
        super(GCN_sim, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.neighbor_k = neighbor_k
        self.gc1 = GraphConvolution(d1, d2)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()

        self.fc_layer = nn.Linear(d1, d1//4)
        self.norm_1 = nn.LayerNorm(d1)

    def forward(self, X, fc=False, test=False):
        '''
        input X: N, d
        output X: d
        '''
        # X = self.dropout(X)
        # Gsim = self.cul_Gsim(X, front_mask, test)
        # X = self.relu(self.gc1(X, Gsim))

        X_1 = self.norm_1(X)
        Gsim = self.cul_Gsim(X_1, fc, test)
        X_1 = self.gc1(X_1, self.dropout(Gsim))
        X = X + self.dropout(X_1)

        # X_2 = self.norm_2(X)
        # X_2 = self.fc(X_2)
        # X = X + self.dropout(X_2)

        if test:
            return Gsim
            
        return X

    def cul_Gsim(self, X, fc, test):
        # if fc:
        X = self.fc_layer(X)

        X_norm = F.normalize(X, p=2, dim=1)
        g_sim = torch.matmul(X_norm, X_norm.t())

        # if front_mask is not None:
            # front_mask = (1 - front_mask).byte()
            # g_sim.masked_fill_(front_mask, 0)
            # g_sim = g_sim * front_mask
        
        if self.neighbor_k > 0:
            g_sim = self.process_neighbor(g_sim)
        g_sim = F.softmax(g_sim, dim=1)
        return g_sim

    def process_neighbor(self, g_sim):
        N = g_sim.size(1)
        k = self.neighbor_k if N>=self.neighbor_k else N

        _, top_k = g_sim.topk(k)
        mask = torch.ones_like(g_sim).byte().cuda()
        for n in range(N):
            mask[n][top_k[n]] = 0
        g_sim[mask] = -float('Inf')
        return g_sim
    
    def __repr__(self):
        return self.__class__.__name__ + ' N, {} -> N, {}, neighbor number is {}.'.format(self.d1, self.d2, self.neighbor_k)

if __name__ == "__main__":
    import torch

    # hyperparameters
    N = 16*10
    d = 512
    b_z = 32
    class_num = 1000

    # input
    G_front = torch.ones(b_z, N, N)
    G_back = torch.ones(b_z, N, N)
    X = torch.rand(b_z, N, d)  # batch, N, d
    labels = torch.ones(b_z, dtype=torch.long)

    # model
    model = GCN_sim(N, d)
    # model = GCN_front_back(N, d)
    # model.load_state_dict(torch.load('../pretrained/GCN_model.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # GPU
    model.cuda()
    X = X.cuda()
    G_front = G_front.cuda()
    G_back = G_back.cuda()
    labels = labels.cuda()
    fc = torch.nn.Linear(512, 7).cuda()

    # forward
    for _ in range(10):
        model.train()
        optimizer.zero_grad()
        # output = model(X, G_front, G_back)
        output = model(X)
        result = fc(output)

        loss_train = F.cross_entropy(result, labels)

        loss_train.backward()
        optimizer.step()
        print('step')
    
        print(model.gc1.weight)

    # for n, p in enumerate(model.parameters()):
    #     if p.requires_grad:
    #         print(n)
    #         print(p)
    #         print(p.grad)

