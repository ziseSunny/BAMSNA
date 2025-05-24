import torch
import torch.nn.functional as F
from loss import feature_reconstruct_loss

def train_wgan_adv_pseudo_self_dual(trans, optimizer_trans, wdiscriminator, optimizer_d, networks, lambda_gp=10,
                                    batch_d_per_iter=5, batch_size_align=512):
    models = [t[0] for t in networks]
    features = [t[2] for t in networks]
    edges = [t[3] for t in networks]

    embd0_lr = models[0](features[0], edges[0])
    embd1_lr = trans(models[1](features[1], edges[1]))

    embd0_rl = models[1](features[1], edges[1])
    embd1_rl = trans(models[0](features[0], edges[0]))

    trans.train()
    wdiscriminator.train()
    models[0].train()
    models[1].train()

    for j in range(batch_d_per_iter):
        w0_lr = wdiscriminator(embd0_lr)
        w1_lr = wdiscriminator(embd1_lr)
        anchor1_lr = w1_lr.view(-1).argsort(descending=True)[: embd1_lr.size(0)] # Algorithm 1
        anchor0_lr = w0_lr.view(-1).argsort(descending=False)[: embd1_lr.size(0)]
        embd0_anchor_lr = embd0_lr[anchor0_lr, :].clone().detach()
        embd1_anchor_lr = embd1_lr[anchor1_lr, :].clone().detach()

        loss_lr = -torch.mean(wdiscriminator(embd0_anchor_lr)) + torch.mean(wdiscriminator(embd1_anchor_lr))

        w0_rl = wdiscriminator(embd0_rl)
        w1_rl = wdiscriminator(embd1_rl)
        anchor1_rl = w1_rl.view(-1).argsort(descending=True)[: embd1_rl.size(0)]
        anchor0_rl = w0_rl.view(-1).argsort(descending=False)[: embd1_rl.size(0)]
        embd0_anchor_rl = embd0_rl[anchor0_rl, :].clone().detach()
        embd1_anchor_rl = embd1_rl[anchor1_rl, :].clone().detach()

        loss_rl = -torch.mean(wdiscriminator(embd0_anchor_rl)) + torch.mean(wdiscriminator(embd1_anchor_rl))

        optimizer_d.zero_grad()

        loss = loss_lr + loss_rl

        loss.backward()
        optimizer_d.step()
        for p in wdiscriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
    w0_lr = wdiscriminator(embd0_lr)
    w1_lr = wdiscriminator(embd1_lr)
    anchor1_lr = w1_lr.view(-1).argsort(descending=True)[: embd1_lr.size(0)]
    anchor0_lr = w0_lr.view(-1).argsort(descending=False)[: embd1_lr.size(0)]
    embd0_anchor_lr = embd0_lr[anchor0_lr, :]
    embd1_anchor_lr = embd1_lr[anchor1_lr, :]
    loss_lr = -torch.mean(wdiscriminator(embd1_anchor_lr))

    w0_rl = wdiscriminator(embd0_rl)
    w1_rl = wdiscriminator(embd1_rl)
    anchor1_rl = w1_rl.view(-1).argsort(descending=True)[: embd1_rl.size(0)]
    anchor0_rl = w0_rl.view(-1).argsort(descending=False)[: embd1_rl.size(0)]
    embd0_anchor_rl = embd0_rl[anchor0_rl, :]
    embd1_anchor_rl = embd1_rl[anchor1_rl, :]
    loss_rl = -torch.mean(wdiscriminator(embd1_anchor_rl))

    loss = loss_lr + loss_rl

    return loss


def train_feature_recon(trans, optimizer_trans, networks, recon_models, optimizer_recons, batch_r_per_iter=10):
    models = [t[0] for t in networks]
    features = [t[2] for t in networks]
    edges = [t[3] for t in networks]
    recon_model0, recon_model1 = recon_models
    optimizer_recon0, optimizer_recon1 = optimizer_recons
    embd0 = models[0](features[0], edges[0])
    embd1 = trans(models[1](features[1], edges[1]))

    recon_model0.train()
    recon_model1.train()
    trans.train()
    models[0].train()
    models[1].train()
    embd0_copy = embd0.clone().detach()
    embd1_copy = embd1.clone().detach()
    for t in range(batch_r_per_iter):
        optimizer_recon0.zero_grad()
        loss = feature_reconstruct_loss(embd0_copy, features[0], recon_model0)
        loss.backward()
        optimizer_recon0.step()
    for t in range(batch_r_per_iter):
        optimizer_recon1.zero_grad()
        loss = feature_reconstruct_loss(embd1_copy, features[1], recon_model1)
        loss.backward()
        optimizer_recon1.step()
    loss = 0.5 * feature_reconstruct_loss(embd0, features[0], recon_model0) + 0.5 * feature_reconstruct_loss(embd1, features[1], recon_model1)

    return loss


def get_hits(embds, ground_truth, k=50, mode='cosine', prior=None, prior_rate=0):
    embd0, embd1 = embds[0].clone().detach(), embds[1].clone().detach()
    g_map_rl = {}
    for i in range(ground_truth.size(1)):
        g_map_rl[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list_rl = list(g_map_rl.keys())

    cossim = torch.zeros(embd1.size(0), embd0.size(0), device='cuda')
    for i in range(embd1.size(0)):
        cossim[i] = F.cosine_similarity(embd0, embd1[i:i + 1].expand(embd0.size(0), embd1.size(1)), dim=-1).view(-1)
    if prior is not None:
        cossim = (1 + cossim) / 2 * (1 - prior_rate) + prior * prior_rate

    ind_rl = cossim.argsort(dim=1, descending=True)
    a1_rl = 0
    a5_rl = 0
    a10_rl = 0
    a20_rl = 0
    a30_rl = 0
    a40_rl = 0
    ak_rl = 0
    mrr_rl = 0
    auc_rl = 0
    for i, node in enumerate(g_list_rl):
        if ind_rl[node, 0].item() == g_map_rl[node]:
            a1_rl += 1
            a5_rl += 1
            a10_rl += 1
            a20_rl += 1
            a30_rl += 1
            a40_rl += 1
            ak_rl += 1
            mrr_rl += 1.0
            auc_rl += 1.0
        else:
            for j in range(1, ind_rl.shape[1]):
                if ind_rl[node, j].item() == g_map_rl[node]:
                    if j < 5:
                        a5_rl += 1
                    if j < 10:
                        a10_rl += 1
                    if j < 20:
                        a20_rl += 1
                    if j < 30:
                        a30_rl += 1
                    if j < 40:
                        a40_rl += 1
                    if j < k:
                        ak_rl += 1
                    mrr_rl += 1.0 / (j + 1)
                    auc_rl += (ind_rl.shape[1] - (j + 1)) / (ind_rl.shape[1] - 1)
                    break
    a1_rl /= len(g_list_rl)
    a5_rl /= len(g_list_rl)
    a10_rl /= len(g_list_rl)
    a20_rl /= len(g_list_rl)
    a30_rl /= len(g_list_rl)
    a40_rl /= len(g_list_rl)
    ak_rl /= len(g_list_rl)
    mrr_rl /= len(g_list_rl)
    auc_rl /= len(g_list_rl)

    g_map_lr = {}
    for i in range(ground_truth.size(1)):
        g_map_lr[ground_truth[0, i].item()] = ground_truth[1, i].item()
    g_list_lr = list(g_map_lr.keys())

    ind_lr = cossim.t().argsort(dim=1, descending=True)
    a1_lr = 0
    a5_lr = 0
    a10_lr = 0
    a20_lr = 0
    a30_lr = 0
    a40_lr = 0
    ak_lr = 0
    mrr_lr = 0
    auc_lr = 0
    for i, node in enumerate(g_list_lr):
        if ind_lr[node, 0].item() == g_map_lr[node]:
            a1_lr += 1
            a5_lr += 1
            a10_lr += 1
            a20_lr += 1
            a30_lr += 1
            a40_lr += 1
            ak_lr += 1
            mrr_lr += 1.0
            auc_lr += 1.0
        else:
            for j in range(1, ind_lr.shape[1]):
                if ind_lr[node, j].item() == g_map_lr[node]:
                    if j < 5:
                        a5_lr += 1
                    if j < 10:
                        a10_lr += 1
                    if j < 20:
                        a20_lr += 1
                    if j < 30:
                        a30_lr += 1
                    if j < 40:
                        a40_lr += 1
                    if j < k:
                        ak_lr += 1
                    mrr_lr += 1.0 / (j + 1)
                    auc_lr += (ind_lr.shape[1] - (j + 1)) / (ind_lr.shape[1] - 1)
                    break
    a1_lr /= len(g_list_lr)
    a5_lr /= len(g_list_lr)
    a10_lr /= len(g_list_lr)
    a20_lr /= len(g_list_lr)
    a30_lr /= len(g_list_lr)
    a40_lr /= len(g_list_lr)
    ak_lr /= len(g_list_lr)
    mrr_lr /= len(g_list_lr)
    auc_lr /= len(g_list_lr)

    print('right2left: H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%%' % (
        a1_rl * 100, a5_rl * 100, a10_rl * 100, a20_rl * 100, a30_rl * 100, a40_rl * 100, ak_rl * 100, mrr_rl * 100, auc_rl * 100))

    print('left2right: H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%%' % (
        a1_lr * 100, a5_lr * 100, a10_lr * 100, a20_lr * 100, a30_lr * 100, a40_lr * 100, ak_lr * 100, mrr_lr * 100, auc_lr * 100))

    a1 = (a1_rl + a1_lr) / 2
    a5 = (a5_rl + a5_lr) / 2
    a10 = (a10_rl + a10_lr) / 2
    a20 = (a20_rl + a20_lr) / 2
    a30 = (a30_rl + a30_lr) / 2
    a40 = (a40_rl + a40_lr) / 2
    ak = (ak_rl + ak_lr) / 2
    mrr = (mrr_rl + mrr_lr) / 2
    auc = (auc_rl + auc_lr) / 2

    print('average: H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@20 %.2f%% H@30 %.2f%% H@40 %.2f%% H@50 %.2f%% MRR %.2f%% AUC %.2f%%' % (
        a1 * 100, a5 * 100, a10 * 100, a20 * 100, a30 * 100, a40 * 100, ak * 100, mrr * 100, auc * 100))

    return a1, a5, a10, a20, a30, a40, ak, mrr, auc