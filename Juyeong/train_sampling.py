import copy
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from config import CONFIG
from modules import GCNNet
from utils import evaluate, load_data, calc_f1
import warnings

from aug import HLoss, Jensen_Shannon, generate_aug_graph

def main(args, task):
    warnings.filterwarnings('ignore')
    multilabel_data = {'ppi', 'yelp', 'amazon'}
    multilabel = args.dataset in multilabel_data

    # This flag is excluded for too large dataset, like amazon, the graph of which is too large to be directly
    # shifted to one gpu. So we need to
    # 1. put the whole graph on cpu, and put the subgraphs on gpu in training phase
    # 2. put the model on gpu in training phase, and put the model on cpu in validation/testing phase
    # We need to judge cpu_flag and cuda (below) simultaneously when shift model between cpu and gpu
    if args.dataset in ['amazon']:
        cpu_flag = True
    else:
        cpu_flag = False

    # load and preprocess dataset
    data = load_data(args, multilabel)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = data.train_nid

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    # load sampler

    kwargs = {
        'dn': args.dataset, 'g': g, 'train_nid': train_nid, 'num_workers_sampler': args.num_workers_sampler,
        'num_subg_sampler': args.num_subg_sampler, 'batch_size_sampler': args.batch_size_sampler,
        'online': args.online, 'num_subg': args.num_subg, 'full': args.full}

    if args.sampler == "node":
        saint_sampler = SAINTNodeSampler(args.node_budget, **kwargs)
    elif args.sampler == "edge":
        saint_sampler = SAINTEdgeSampler(args.edge_budget, **kwargs)
    elif args.sampler == "rw":
        saint_sampler = SAINTRandomWalkSampler(args.num_roots, args.length, **kwargs)
    else:
        raise NotImplementedError

    loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if not cpu_flag:
            g = g.to('cuda:{}'.format(args.gpu))

    print('labels shape:', g.ndata['label'].shape)
    print("features shape:", g.ndata['feat'].shape)

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        aggr=args.aggr
    )

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("GPU memory allocated before training(MB)",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1

    subg_t = [[] for _ in range(2)]
    # subg_t = torch.empty((1, 2), dtype=torch.int32).to(torch.cuda.current_device())

    h_loss_op = HLoss()
    js_loss_op = Jensen_Shannon()

    for epoch in range(args.n_epochs):
        for j, subg in enumerate(loader):
            if cuda:
                subg = subg.to(torch.cuda.current_device())
            # Augment Subgraph
            if epoch == 0:
                subg_t[0].append(copy.deepcopy(subg))
            else:
                subg_t[0].append(copy.deepcopy(subg_t[1][j])) 
                ### ???
                # subg_t[0] = copy.deepcopy(subg_t[1][j]) 

            ### Shouldn't subg_t[0][j] be subg_t[0][-1][j] ???
            auged_subg, delta_G_e, delta_G_v, delta_G_e_aug, delta_G_v_aug \
                = generate_aug_graph(subg_t[0][j], model,
                                     args.sigma_delta_e, args.sigma_delta_v, args.mu_e, args.mu_v,
                                     args.lam1_e, args.lam1_v, args.lam2_e, args.lam2_v,
                                     args.a_e, args.b_e, args.a_v, args.b_v)
            subg_t[1].append(auged_subg)

            # Start Training
            model.train()

            # forward
            pred = model(subg)
            pred1 = model(subg_t[0][j])
            pred2 = model(subg_t[1][j])

            # Calculate loss
            # if multilabel:
            #     loss_XE = F.binary_cross_entropy_with_logits(pred, subg.ndata['label'], reduction='sum',
            #                                                  weight=subg.ndata['l_n'].unsqueeze(1))
            # else:
            #     loss_XE = F.cross_entropy(pred, subg.ndata['label'], reduction='none')
            #     loss_XE = (subg.ndata['l_n'] * loss_XE).sum()

            if multilabel:
                loss_XE = F.binary_cross_entropy_with_logits(pred1, subg_t[0][j].ndata['label'], reduction='sum',
                                                             weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss_XE = F.cross_entropy(pred1, subg_t[0][j].ndata['label'], reduction='none')
                loss_XE = (subg_t[0][j].ndata['l_n'] * loss_XE).sum()

            if delta_G_e + delta_G_v < delta_G_e_aug + delta_G_v_aug:
                loss_KL = js_loss_op(pred1.detach(), pred2)
            else:
                loss_KL = js_loss_op(pred1, pred2.detach())

            loss_H = h_loss_op(pred)

            total_loss = (loss_XE +
                          args.kl * loss_KL +
                          args.h * loss_H)

            # total_loss = loss_XE

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            if j == len(loader) - 1:
                model.eval()
                with torch.no_grad():
                    train_f1_mic, train_f1_mac = calc_f1(subg.ndata['label'].cpu().numpy(),
                                                         pred.cpu().numpy(), multilabel)
                    print(f"epoch:{epoch + 1}/{args.n_epochs}, Iteration {j + 1}/"
                          f"{len(loader)}:training loss", total_loss.item())
                    print("Train F1-mic {:.4f}, Train F1-mac {:.4f}".format(train_f1_mic, train_f1_mac))
        # evaluate
        model.eval()
        if epoch % args.val_every == 0:
            if cpu_flag and cuda:  # Only when we have shifted model to gpu and we need to shift it back on cpu
                model = model.to('cpu')
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels.cpu(), val_mask.cpu(), multilabel)
            print(
                "Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
            if cpu_flag and cuda:
                model.cuda()

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    # test
    if cpu_flag and cuda:
        model = model.to('cpu')
    test_f1_mic, test_f1_mac = evaluate(
        model, g, labels.cpu(), test_mask.cpu(), multilabel)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='GraphSAINT')
    # parser.add_argument("--task", type=str, default="ppi_n", help="type of tasks")
    parser.add_argument("--task", type=str, default="flickr_n", help="type of tasks")
    parser.add_argument("--online", dest='online', action='store_true', help="sampling method in training phase")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    task = parser.parse_args().task
    args = argparse.Namespace(**CONFIG[task])
    args.online = parser.parse_args().online
    args.gpu = parser.parse_args().gpu
    if args.dataset == 'yelp' or args.dataset == 'reddit':
        torch.multiprocessing.set_sharing_strategy('file_system')
    print(args)

    main(args, task=task)
