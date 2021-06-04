import torch


def ONInvariantMap(xs):
    out = torch.zeros_like(xs)
    out[:, 0] = torch.norm(xs, dim=1)
    return out


def ENInvariantMap(xs):
    def center_around_0(inp):
        # Center around 0 for translation invariance
        CoM = torch.mean(inp, dim=1)
        centered = inp - CoM[:, None, :]
        return centered

    def order(centered):
        # Order by radius to make permutation invariant
        r = torch.sqrt(centered[:, :, 0] ** 2 + centered[:, :, 1] ** 2)
        order = torch.argsort(r)
        sorted = centered[torch.arange(order.size(0)).unsqueeze(1).repeat((1, order.size(1))), order]
        return sorted

    def rotate(centered, sorted):
        r = torch.sqrt(centered[:, :, 0] ** 2 + centered[:, :, 1] ** 2)
        # Rotate to x axis
        top_1 = sorted[:, -1]
        top_r_1 = r.max(dim=1)[0]
        norm_1 = top_1 / top_r_1[:, None]
        t0 = torch.cat([norm_1[:, 0, None], -norm_1[:, 1, None]], dim=1).T
        t1 = torch.cat([norm_1[:, 1, None], norm_1[:, 0, None]], dim=1).T
        G_1 = torch.stack([t0, t1]).permute([2, 1, 0])[:, None, :, :]
        rotated_1 = torch.matmul(G_1, sorted[:, :, :, None]).squeeze()
        return rotated_1

    inp = xs.view(-1, 4, 2)
    centered = center_around_0(inp)
    sorted = order(centered)
    rotated = rotate(centered, sorted)

    s_1 = sorted.size()
    input = rotated.view((s_1[0], s_1[1] * s_1[2]))

    return input
