import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    # 只更新weight，不更新bias和BN等参数
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean,targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        out = self.proxy(inputs_adv)
        if isinstance(out, tuple):
            # print("dual predict")
            adv_pred, adv_feature = out
            loss = F.cross_entropy(adv_pred, targets)
            
            clean_pred, clean_feature = self.proxy(inputs_clean)

            normed_clean = F.normalize(clean_feature, dim=-1)
            matrix_clean = torch.mm(normed_clean, normed_clean.t())

            normed_feature = F.normalize(adv_feature, dim = -1)
            matrix_adv = torch.mm(normed_feature, normed_feature.t())
            
            diff = torch.exp(torch.abs(matrix_adv - matrix_clean))
            loss_sp = 10*torch.mean(diff)

            loss += loss_sp
            loss = -1.0*loss

        else:
            print("single predict")
            loss = - F.cross_entropy(out, targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    #对权重进行扰动
    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    # 权重还要变回来
    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




