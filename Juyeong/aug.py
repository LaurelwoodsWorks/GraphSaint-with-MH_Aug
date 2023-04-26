import copy
import random
import numpy as np
import scipy
import torch as th
import dgl

# if __name__ == '__main__':
#     from modules import AGGNet
# else:
#     from Juyeong.modules import AGGNet
from modules import AGGNet
from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, full=False):
        num_data = x.shape[0]
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if full:
            return -1.0 * b.sum(1)
        b = -1.0 * b.sum()
        b = b / num_data
        return b


class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon, self).__init__()

    def forward(self, y, x):
        num_data = x.shape[0]
        b = F.softmax(y, dim=1) * F.log_softmax(x, dim=1) - F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        b += F.softmax(x, dim=1) * F.log_softmax(y, dim=1) - F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -0.5 * b.sum()
        b = b / num_data
        return b


def our_truncnorm(a, b, mu, sigma, x=None, mode='pdf'):
    a, b = (a - mu) / sigma, (b - mu) / sigma
    if mode=='pdf':
        return truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    elif mode=='rvs':
        return truncnorm.rvs(a, b, loc = mu, scale = sigma)

def aggregate(graph, agg_model):
    s_vec = agg_model(graph)
    return s_vec

def log_normal(a, b, sigma):
    return -1 * th.pow(a - b, 2) / (2 * th.pow(sigma, 2)) #/root2pi / sigma

def augment(g, delta_G_e, delta_G_v):
    num_edge_drop = int(g.num_edges() * delta_G_e)
    idx = th.randperm(num_edge_drop, device='cuda:0')[num_edge_drop:]
    g.remove_edges(idx)

    n = g.num_nodes()
    num_node_drop = int(n * delta_G_v)
    aug_feature = g.ndata['feat']
    node_list = th.ones(n, 1, device='cuda:0')
    idx = th.randperm(n, device='cuda:0')[:num_node_drop]
    aug_feature[idx] = 0
    node_list[idx] = 0
    if num_node_drop:
        aug_feature *= n / (n - num_node_drop)
    g.ndata['feat'] = aug_feature

    return g, node_list


def generate_aug_graph(g, model,
                       sigma_delta_e=0.03, sigma_delta_v=0.03, mu_e=0.6, mu_v=0.2,
                       lam1_e=1, lam1_v=1, lam2_e=0.0, lam2_v=0.0,
                       a_e=100, b_e=1, a_v=100, b_v=1):
    # Original Graph Feature and Metadata Extraction, Preprocessing
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    coo_mat = g.edges(form='uv')
    coo_mat = th.tensor([list(coo_mat[0]), list(coo_mat[1])], device='cuda:0')
    n_list = th.ones(num_nodes)

    # Create Aggregate Model
    agg_model = AGGNet(num_hop=2)
    agg_model.cuda()

    while True:
        # Calculate Delta Value
        delta_G_e = 1 - coo_mat.shape[1] / num_edges
        delta_G_e_aug = our_truncnorm(0, 1, delta_G_e, sigma_delta_e, mode='rvs')

        delta_G_v = 1 - n_list.sum().item() / num_nodes
        delta_G_v_aug = our_truncnorm(0, 1, delta_G_v, sigma_delta_v, mode='rvs')

        # Graph Augmentation According To Delta Value
        aug_g, aug_n_list = augment(g, delta_G_e_aug, delta_G_v_aug)
        aug_g = dgl.add_self_loop(aug_g)

        # message_passing_g = copy.deepcopy(g)
        message_passing_g = g.clone()
        message_passing_g.ndata['feat'] = th.ones(num_nodes, 1, device='cuda:0')

        # message_passing_aug_g = copy.deepcopy(aug_g)
        message_passing_aug_g = aug_g.clone()
        message_passing_aug_g.ndata['feat'] = th.ones(num_nodes, 1, device='cuda:0')

        # Calculate ego-graph's message passing value
        with th.no_grad():
            org_ego = aggregate(message_passing_g, agg_model)

        # Calculate Augmented Delta Value
        with th.no_grad():
            delta_g_e = 1 - (aggregate(message_passing_g, agg_model) / org_ego).squeeze(1)
            delta_g_aug_e = 1 - (aggregate(message_passing_aug_g, agg_model) / org_ego).squeeze(1)
            delta_g_v = 1 - (aggregate(message_passing_g, agg_model) / org_ego).squeeze(1)
            delta_g_aug_v = 1 - (aggregate(message_passing_aug_g, agg_model) / org_ego).squeeze(1)

        # Calculate Target Distribution and Proposal Distribution
        h_loss_op = HLoss()

        with th.no_grad():
            output = model(g)

        max_ent = h_loss_op(th.full((1, output.shape[1]), 1 / output.shape[1])).item()
        ent = h_loss_op(output.detach(), True) / max_ent
        
        # log_normal: normal distribution에 log를 취한 것
        p = lam1_e * log_normal(delta_g_e, mu_e, a_e * ent + b_e) + \
            0
            # lam1_v * log_normal(delta_g_v, mu_v, a_v * ent + b_v)
        p_aug = lam1_e * log_normal(delta_g_aug_e, mu_e, a_e * ent + b_e) + \
            0
            # lam1_v * log_normal(delta_g_aug_v, mu_v, a_v * ent + b_v)

        q = np.log(our_truncnorm(0, 1, delta_G_e_aug, sigma_delta_e, x=delta_G_e, mode='pdf')) + \
            lam2_e * scipy.special.betaln(num_edges - num_edges * delta_G_e + 1, num_edges * delta_G_e + 1) + \
            np.log(our_truncnorm(0, 1, delta_G_v_aug, sigma_delta_v, x=delta_G_v, mode='pdf')) + \
            lam2_v * scipy.special.betaln(num_nodes - num_nodes * delta_G_v + 1, num_nodes * delta_G_v + 1)
        q_aug = np.log(our_truncnorm(0, 1, delta_G_e, sigma_delta_e, x=delta_G_e_aug, mode='pdf')) + \
            lam2_e * scipy.special.betaln(num_edges - num_edges * delta_G_e_aug + 1,
                                          num_edges * delta_G_e_aug + 1) + \
            np.log(our_truncnorm(0, 1, delta_G_v, sigma_delta_v, x=delta_G_v_aug, mode='pdf')) + \
            lam2_v * scipy.special.betaln(num_nodes - num_nodes * delta_G_v_aug + 1, num_nodes * delta_G_v_aug + 1)

        # Calculate Acceptance
        acceptance = ((th.sum(p_aug) - th.sum(p)) - (q_aug - q))
        if np.log(random.random()) < acceptance:
            break
    return aug_g, delta_G_e, delta_G_v, delta_G_e_aug, delta_G_v_aug
