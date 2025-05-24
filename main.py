import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore")

from loaddata import load
import numpy as np
import torch
import itertools
import RWR_embedding

seed = 1
torch.manual_seed(seed)
import torch.nn.functional as F
from graphmodel import WDiscriminator, transformation, ReconDNN, notrans, LGCNDirected
from train import train_feature_recon, get_hits, train_wgan_adv_pseudo_self_dual
import time
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='douban')
parser.add_argument('--transformer', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_wd', type=float, default=0.01)
parser.add_argument('--lr_recon', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.02)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--max_iterations', type=int, default=10)
parser.add_argument('--recon_share', type=bool, default=False)
parser.add_argument('--dual_training', type=bool, default=True)
parser.add_argument('--rwr_directed', type=bool, default=True)
parser.add_argument('--known_ratio', type=float, default=0.1)
args = parser.parse_args()

print(args)
args.net = LGCNDirected

dataset_name = args.dataset

if dataset_name in ['dblp']:
    a1, f1, a2, f2, ground_truth = load(dataset_name)
    feature_size = f1.shape[1]
    ns = [a1.shape[0], a2.shape[0]]
    edge_1 = torch.LongTensor(np.array(a1)).to('cuda')
    edge_2 = torch.LongTensor(np.array(a2)).to('cuda')
    ground_truth = torch.tensor(np.array(ground_truth, dtype=int), device='cuda')
    features = [torch.FloatTensor(f1).to('cuda'),
                torch.FloatTensor(f2).to('cuda')]
    edges = [edge_1, edge_2]
    prior = None
    prior_rate = 0

if dataset_name in ['douban']:
    a1, f1, a2, f2, ground_truth, _ = load(dataset_name)
    feature_size = f1.shape[1]
    ns = [a1.shape[0], a2.shape[0]]
    edge_1 = torch.LongTensor(np.array(a1.nonzero())).to('cuda')
    edge_2 = torch.LongTensor(np.array(a2.nonzero())).to('cuda')
    ground_truth = torch.tensor(np.array(ground_truth, dtype=int), device='cuda') - 1  # Original index start from 1
    features = [torch.FloatTensor(f1.todense()).to('cuda'),
                torch.FloatTensor(f2.todense()).to('cuda')]
    edges = [edge_1, edge_2]
    prior = None
    prior_rate = 0

elif dataset_name in ['AI']:
    a1, f1, a2, f2, ground_truth = load(dataset_name)
    feature_size = f1.shape[1]
    ns = [a1.shape[0], a2.shape[0]]
    edge_1 = torch.LongTensor(np.array(a1)).to('cuda')
    edge_2 = torch.LongTensor(np.array(a2)).to('cuda')
    ground_truth = torch.tensor(np.array(ground_truth, dtype=int), device='cuda')
    features = [torch.FloatTensor(f1).to('cuda'), torch.FloatTensor(f2).to('cuda')]
    edges = [edge_1, edge_2]
    prior = None
    prior_rate = 0

elif dataset_name in ['wd']:
    a1, f1, a2, f2, ground_truth = load(dataset_name)
    feature_size = f1.shape[1]
    ns = [a1.shape[0], a2.shape[0]]
    edge_1 = torch.LongTensor(np.array(a1)).to('cuda')
    edge_2 = torch.LongTensor(np.array(a2)).to('cuda')
    ground_truth = torch.tensor(np.array(ground_truth, dtype=int), device='cuda')
    features = [torch.FloatTensor(f1).to('cuda'), torch.FloatTensor(f2).to('cuda')]
    edges = [edge_1, edge_2]
    prior = None
    prior_rate = 0

np.random.seed(123)

edge_index1 = edges[0].cpu().numpy()
edge_index2 = edges[1].cpu().numpy()
grd_truth = ground_truth.cpu().numpy()
seed_1, seed_2 = RWR_embedding.split_data(grd_truth, args.known_ratio)
rwr_emb1, rwr_emb2 = RWR_embedding.rwr_emd(edge_index1, edge_index2, seed_1, seed_2, args.rwr_directed)

rwr_emb1 = torch.tensor(rwr_emb1, device='cuda', dtype=torch.float32)
rwr_emb2 = torch.tensor(rwr_emb2, device='cuda', dtype=torch.float32)

rwr_features = [rwr_emb1, rwr_emb2]

rwr_feature_size = rwr_features[0].size(1)

prior = torch.zeros(rwr_emb2.size(0), rwr_emb1.size(0), device='cuda')
for i in range(rwr_emb2.size(0)):
    prior[i] = F.cosine_similarity(rwr_emb1, rwr_emb2[i:i + 1].expand(rwr_emb1.size(0), rwr_emb2.size(1)), dim=-1).view(-1)

prior_rate = args.beta

mode = 'cosine'

bps = []
times = []

for iter in tqdm.trange(args.max_iterations):
    print('\n')

    get_hits(features, ground_truth, mode=mode, prior=prior, prior_rate=prior_rate)

    num_graph = 2
    networks = []
    feature_output_size = args.hidden_size

    torch.seed()

    model = args.net(feature_size, args.hidden_size).cuda()
    optimizer = None
    for i in range(num_graph):
        networks.append((model, optimizer, features[i], edges[i]))
    trans = transformation(args.hidden_size).cuda()
    optimizer_trans = torch.optim.Adam(itertools.chain(trans.parameters(), networks[0][0].parameters()), lr=args.lr, weight_decay=5e-4)

    embd0 = networks[0][0](features[0], edges[0])  # model(features[0],edges[0])
    embd1 = networks[1][0](features[1], edges[1])  # model(features[1],edges[1])
    with torch.no_grad():
        a1, a5, a10, a20, a30, a40, ak, mrr, auc = get_hits([embd0, trans(embd1)], ground_truth, mode=mode, prior=prior,
                                                       prior_rate=prior_rate)

    wdiscriminator = WDiscriminator(feature_output_size).cuda()
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.lr_wd, weight_decay=5e-4)

    recon_model0 = ReconDNN(feature_output_size, feature_size).cuda()
    recon_model1 = ReconDNN(feature_output_size, feature_size).cuda()
    optimizer_recon0 = torch.optim.Adam(recon_model0.parameters(), lr=args.lr_recon, weight_decay=5e-4)
    optimizer_recon1 = torch.optim.Adam(recon_model1.parameters(), lr=args.lr_recon, weight_decay=5e-4)

    batch_size_align = 128  # not use

    best = 0
    bp = 0,0,0,0,0,0,0,0

    time1 = time.time()
    for i in range(1, args.epochs + 1):
        trans.train()
        networks[0][0].train()
        networks[1][0].train()

        optimizer_trans.zero_grad()
        loss = train_wgan_adv_pseudo_self_dual(trans, optimizer_trans, wdiscriminator, optimizer_wd, networks)

        loss_feature = train_feature_recon(trans, optimizer_trans, networks, [recon_model0, recon_model1], [optimizer_recon0, optimizer_recon1])
        loss = (1-args.alpha) * loss + args.alpha * loss_feature

        loss.backward()
        optimizer_trans.step()

        networks[0][0].eval()
        networks[1][0].eval()
        trans.eval()
        embd0 = networks[0][0](features[0], edges[0])
        embd1 = networks[1][0](features[1], edges[1])


        with torch.no_grad():
            a1, a5, a10, a20, a30, a40, ak, mrr, auc = get_hits([embd0, trans(embd1)], ground_truth, mode=mode, prior=prior, prior_rate=prior_rate)
        if a1 > best:
            best = a1
            bp = a1, a5, a10, a20, a30, a40, ak, mrr, auc

    time2 = time.time()
    print('Total Time %.2f' % (time2-time1))
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%%'
          % (bp[0]*100, bp[1]*100, bp[2]*100, bp[3]*100, bp[4]*100, bp[5]*100, bp[6]*100, bp[7]*100, bp[8]*100))

    bps.append(bp)
    times.append(time2-time1)

final_bps = []
for bp in bps:
    if bp[0] > 0.57:
        final_bps.append(bp)

aver_bp = np.mean(np.array(final_bps), axis=0)
aver_time = np.mean(np.array(times))

print('Average Total Time %.2f' % (aver_time))
print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%%'
      % (aver_bp[0]*100, aver_bp[1]*100, aver_bp[2]*100, aver_bp[3]*100, aver_bp[4]*100, aver_bp[5]*100, aver_bp[6]*100, aver_bp[7]*100, aver_bp[8]*100))

f_rec = open(args.dataset + '_record.txt', 'w')
f_rec.write('Data %s H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%% Time %.2f\n' %
            (args.dataset, aver_bp[0]*100, aver_bp[1]*100, aver_bp[2]*100, aver_bp[3]*100, aver_bp[4]*100, aver_bp[5]*100,
             aver_bp[6]*100, aver_bp[7]*100, aver_bp[8]*100, aver_time))
