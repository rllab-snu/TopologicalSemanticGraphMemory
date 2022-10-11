
def compute_spl(gt_dist, executed_action, threshs_c, threshs_m):
    # Find where the agent chose to stop.
    executed_action_ = np.pad(executed_action, ([0, 0], [0, 1]), 'constant')
    episode_len = [np.where(_ == 0)[0][0] for _ in executed_action_]
    episode_len = np.array(episode_len)
    # Clip episode to maximum length.
    episode_len = np.minimum(episode_len, gt_dist.shape[1] - 1)

    # Compute the length that the agent moved.
    d_episode_end, trans_steps = [], []
    for i in range(gt_dist.shape[0]):
        d_episode_end.append(gt_dist[i, episode_len[i]])
        trans_steps.append(np.sum(executed_action_[i, :(1 + episode_len[i])] == 3))
    trans_steps = np.array(trans_steps)
    d_episode_end = np.array(d_episode_end)
    d_start = gt_dist[:, 0]

    # Function to compute the SPL metric.
    spls = []
    all_spls = []
    for s_c, s_m in zip(threshs_c, threshs_m):
        success = d_episode_end <= np.maximum(s_c, s_m * d_start)
        success = success * 1.
        spl = success * (d_start / np.maximum(d_start, trans_steps * 8.))
        all_spls.append(spl)
        spls.append(np.mean(spl))
    return spls, all_spls



def voc_ap_fast(rec, prec):
    rec = rec.reshape((-1, 1))
    prec = prec.reshape((-1, 1))
    z = np.zeros((1, 1))
    o = np.ones((1, 1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))
    mpre_ = np.maximum.accumulate(mpre[::-1])[::-1]
    I = np.where(mrec[1:] != mrec[0:-1])[0] + 1;
    ap = np.sum((mrec[I] - mrec[I - 1]) * mpre[I])
    return np.array(ap).reshape(1, )


def voc_ap(rec, prec):
    rec = rec.reshape((-1, 1))
    prec = prec.reshape((-1, 1))
    z = np.zeros((1, 1))
    o = np.ones((1, 1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = np.where(mrec[1:] != mrec[0:-1])[0] + 1;
    ap = 0;
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i];
    return ap



def calc_pr(gt, out, wt=None, fast=False):
    """Computes VOC 12 style AP (dense sampling).
    returns ap, rec, prec"""
    if wt is None:
        wt = np.ones((gt.size, 1))

    gt = gt.astype(np.float64).reshape((-1, 1))
    wt = wt.astype(np.float64).reshape((-1, 1))
    out = out.astype(np.float64).reshape((-1, 1))

    gt = gt * wt
    tog = np.concatenate([gt, wt, out], axis=1) * 1.
    ind = np.argsort(tog[:, 2], axis=0)[::-1]
    tog = tog[ind, :]
    cumsumsortgt = np.cumsum(tog[:, 0])
    cumsumsortwt = np.cumsum(tog[:, 1])
    prec = cumsumsortgt / cumsumsortwt
    rec = cumsumsortgt / np.sum(tog[:, 0])

    if fast:
        ap = voc_ap_fast(rec, prec)
    else:
        ap = voc_ap(rec, prec)
    return ap, rec, prec
