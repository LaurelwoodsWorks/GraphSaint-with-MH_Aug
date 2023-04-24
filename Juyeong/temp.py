import random
import numpy as np
import scipy
import torch as th
import dgl

import aug
from modules import AGGNet


def generate_aug_graph(g, model, sigma_delta_e=0.03, sigma_delta_v=0.03, mu_e=0.6, mu_v=0.2,
                       lam1_e=1, lam1_v=1, lam2_e=0.0, lam2_v=0.0,
                       a_e=100, b_e=1, a_v=100, b_v=1):
    # Original Graph Feature and Metadata Extraction, Preprocessing
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    coo_mat = g.edges(form='uv')
    n_list = th.ones(num_nodes)
    feats = g.ndata['feats']

    while True:
        # Calculate Delta Value
        delta_G_e = 1 - coo_mat.shape[1] / num_edges
        delta_G_e_aug = aug.our_truncnorm(0, 1, delta_G_e, sigma_delta_e, mode='rvs')

        delta_G_v = 1 - n_list.sum().item() / num_nodes
        delta_G_v_aug = aug.our_truncnorm(0, 1, delta_G_v, sigma_delta_v, mode='rvs')

        # Graph Augmentation According To Delta Value
        aug_coo_mat, aug_feats, aug_n_list = aug.augment(coo_mat, feats, delta_G_e_aug, delta_G_v_aug)
        aug_num_nodes = len(aug_n_list)

        aug_g = dgl.from_scipy(scipy.sparse.coo_matrix((aug_n_list, (aug_coo_mat[0], aug_coo_mat[1])),
                                                       shape=(aug_num_nodes, aug_num_nodes)))

        # Create Aggregate Model
        agg_model = AGGNet(num_hop=2)

        # Calculate ego-graph's message passing value
        with th.no_grad():
            org_ego = aug.aggregate(g, agg_model)

        # Calculate Augmented Delta Value
        with th.no_grad():
            delta_g_e = 1 - (aug.aggregate(g, agg_model) / org_ego).squeeze(1)
            delta_g_aug_e = 1 - (aug.aggregate(aug_g, agg_model) / org_ego).squeeze(1)
            delta_g_v = 1 - (aug.aggregate(g, agg_model) / org_ego).squeeze(1)
            delta_g_aug_v = 1 - (aug.aggregate(aug_g, agg_model) / org_ego).squeeze(1)

        # Calculate Target Distribution and Proposal Distribution
        h_loss_op = aug.HLoss()

        output = model(g)

        max_ent = h_loss_op(th.full((1, output.shape[1]), 1 / output.shape[1])).item()
        ent = h_loss_op(output.detach(), True) / max_ent

        p = lam1_e * aug.log_normal(delta_g_e, mu_e, a_e * ent + b_e) + \
            lam1_v * aug.log_normal(delta_g_v, mu_v, a_v * ent + b_v)
        p_aug = lam1_e * aug.log_normal(delta_g_aug_e, mu_e, a_e * ent + b_e) + \
                lam1_v * aug.log_normal(delta_g_aug_v, mu_v, a_v * ent + b_v)

        q = np.log(aug.our_truncnorm(0, 1, delta_G_e_aug, sigma_delta_e, x=delta_G_e, mode='pdf')) + \
            lam2_e * scipy.special.betaln(num_edges - num_edges * delta_G_e + 1, num_edges * delta_G_e + 1) + \
            np.log(aug.our_truncnorm(0, 1, delta_G_v_aug, sigma_delta_v, x=delta_G_v, mode='pdf')) + \
            lam2_v * scipy.special.betaln(num_nodes - num_nodes * delta_G_v + 1, num_nodes * delta_G_v + 1)
        q_aug = np.log(aug.our_truncnorm(0, 1, delta_G_e, sigma_delta_e, x=delta_G_e_aug, mode='pdf')) + \
            lam2_e * scipy.special.betaln(num_edges - num_edges * delta_G_e_aug + 1,
                                          num_edges * delta_G_e_aug + 1) + \
            np.log(aug.our_truncnorm(0, 1, delta_G_v, sigma_delta_v, x=delta_G_v_aug, mode='pdf')) + \
            lam2_v * scipy.special.betaln(num_nodes - num_nodes * delta_G_v_aug + 1, num_nodes * delta_G_v_aug + 1)

        # Calculate Acceptance
        acceptance = ((th.sum(p_aug) - th.sum(p)) - (q_aug - q))

        if np.log(random.random()) < acceptance:
            break

    return aug_g
