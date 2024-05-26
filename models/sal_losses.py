import numpy as np
import torch
import torch.nn.functional as F


def logit(x):
    return np.log(x / (1 - x + 1e-08) + 1e-08)


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def nss2(s_map, gt):
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    t, w, h = s_map.shape[1:]

    mean_s_map = (
        torch.mean(s_map.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )
    std_s_map = (
        torch.std(s_map.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)


def batch_image_sum(x):
    x = torch.sum(torch.sum(x, 1, keepdim=True), 2, keepdim=True)
    return x


def batch_image_mean(x):
    x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    return x


def cross_entropy_loss(output, label, weights, batch_average=False, is_reduce=True):

    batch_size = output.size(0)
    output = output.reshape(batch_size, -1)
    label = label.reshape(batch_size, -1)

    label = label / 255
    final_loss = F.binary_cross_entropy_with_logits(output, label, reduce=False).sum(1)
    final_loss = final_loss * weights

    if is_reduce:
        final_loss = torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss


def cc_s2(s_map, gt):
    assert s_map.size() == gt.size()
    t, w, h = s_map.shape[1:]
    batch_size = s_map.size(0)

    mean_s_map = (
        torch.mean(s_map.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )
    std_s_map = (
        torch.std(s_map.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )

    mean_gt = (
        torch.mean(gt.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )
    std_gt = (
        torch.std(gt.view(batch_size, -1), 1)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa * bb)))


def kldiv2(s_map, gt):
    assert s_map.size() == gt.size()

    t, w, h = s_map.shape[1:]

    batch_size = s_map.size(0)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1, 1).expand(batch_size, t, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1, 1).expand(batch_size, t, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = torch.tensor(2.2204e-16).to(gt.device)
    result = gt * torch.log(eps + gt / (s_map + eps))
    return torch.mean(torch.sum(result, 1))


def normalize_map2(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    t, w, h = s_map.shape[1:]

    min_s_map = (
        torch.min(s_map.view(batch_size, -1), 1)[0]
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )
    max_s_map = (
        torch.max(s_map.view(batch_size, -1), 1)[0]
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, t, w, h)
    )

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def similarity2(s_map, gt):
    """For single image metric
    Size of Image - WxH or 1xWxH
    gt is ground truth saliency map
    """
    assert s_map.size() == gt.size()
    t, w, h = s_map.shape[1:]
    batch_size = s_map.size(0)

    s_map = normalize_map2(s_map)
    gt = normalize_map2(gt)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1, 1).expand(batch_size, t, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1, 1).expand(batch_size, t, w, h)

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    return torch.mean(torch.sum(torch.min(s_map, gt), 1))


def get_kl_cc_sim_loss(config, pred_map, gt):
    kl_loss = torch.tensor(0.0)
    nss_loss = torch.tensor(0.0)
    cc_loss = torch.tensor(0.0)
    sim_loss = torch.tensor(0.0)

    if config.loss.loss_kl:
        kl_loss = config.loss.kl_weight * kldiv2(pred_map, gt)
    elif config.loss.loss_ce:
        kl_loss = cross_entropy_loss(pred_map, gt, config.loss.ce_weight)
    elif config.loss.loss_mse:
        kl_loss = config.loss.mse_weight * (pred_map - gt).square().sum(
            dim=(1, 2, 3)
        ).mean(dim=0)

    if config.loss.loss_cc:
        cc_loss = config.loss.cc_weight * cc_s2(
            pred_map,
            gt,
        )
    if config.loss.loss_sim:
        sim_loss = config.loss.sim_weight * similarity2(pred_map, gt)
    if config.loss.loss_nss:
        nss_loss = config.loss.nss_weight * nss2(pred_map, gt)

    return kl_loss, cc_loss, sim_loss, nss_loss


def get_kl_cc_sim_loss_wo_weight(config, pred_map, gt):
    kl_loss = torch.tensor(0.0)
    nss_loss = torch.tensor(0.0)
    cc_loss = torch.tensor(0.0)
    sim_loss = torch.tensor(0.0)
    total_loss = torch.tensor(0.0)

    if True and config.loss.loss_kl:
        kl_loss = kldiv2(pred_map, gt)

    if True or config.loss.loss_cc:
        cc_loss = cc_s2(pred_map, gt)
    if True or config.loss.loss_sim:
        sim_loss = similarity2(pred_map, gt)
    if True or config.loss.loss_nss:
        nss_loss = nss2(pred_map, gt)

    # 只能加入这三个的损失函数
    total_loss = nss_loss + cc_loss + sim_loss
    loss = {
        "total": total_loss,
        "main": kl_loss,
        "cc": cc_loss,
        "sim": sim_loss,
        "nss": nss_loss,
    }
    return loss


def get_lossv2(config, predictions, gt):
    total_kl_loss = 0.0
    total_cc_loss = 0.0
    total_sim_loss = 0.0
    total_nss_loss = 0.0
    total_loss = 0.0

    pred_map = predictions
    kl_loss, cc_loss, sim_loss, nss_loss = get_kl_cc_sim_loss(config, pred_map, gt)
    total_kl_loss += kl_loss
    total_cc_loss += cc_loss
    total_sim_loss += sim_loss
    total_nss_loss += nss_loss

    total_loss += kl_loss + cc_loss + sim_loss + nss_loss

    loss = {
        "total": total_loss,
        "main": total_kl_loss,
        "cc": total_cc_loss,
        "sim": total_sim_loss,
        "nss": total_nss_loss,
    }
    return loss
