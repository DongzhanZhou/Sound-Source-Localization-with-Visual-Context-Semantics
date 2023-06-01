import torch
import torch.nn as nn
import torch.nn.functional as F

class CELocLoss(nn.Module):
    def __init__(self, args):
        super(CELocLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.loss_pool = {'maxpool': nn.AdaptiveMaxPool2d((1, 1)), \
                          'avgpool': nn.AdaptiveAvgPool2d((1, 1))}[args.loss_pool]
    def forward(self, A):
        B = A.shape[0]
        logits = self.loss_pool(A).view(B, -1)
        target = torch.arange(B).to(A.device)
        loss = self.criterion(logits, target)
        return loss, logits

def get_div_loss(attmap, avmap, mode):
    B = attmap.shape[0]
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    att_dist = F.softmax(attmap.view(B, -1), dim=1)
    av_dist = F.softmax(avmap.view(B, -1), dim=1)
    if mode == 'kl':
        div_loss = KLDivLoss(att_dist.log(), av_dist)
    elif mode == 'js':
        log_mean = ((att_dist + av_dist) / 2.).log()
        div_loss = (KLDivLoss(log_mean, att_dist) + KLDivLoss(log_mean, av_dist)) / 2.
    else:
        raise NotImplementedError
    return div_loss

class AttCELoss(nn.Module):
    def __init__(self, args):
        super(AttCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.fg_nodes = args.fg_nodes
        self.bg_nodes = args.bg_nodes
        self.loss_names = args.loss_names
        if 'div' in self.loss_names:
            self.div_mode = args.div_mode

    def forward(self, att_feat, aud_feat, att_heatmaps, av_heatmaps):
        B = att_feat.shape[0]
        att_feat = F.normalize(att_feat, dim=1) # [B, num_state, num_node]
        att_sim = torch.einsum('bck,bc->bk', (att_feat, aud_feat)) # [B, num_node]
        sorted_sim, sorted_indexes = torch.sort(att_sim, dim=1, descending=True)
        pos = torch.mean(sorted_sim[:, :self.fg_nodes], dim=1, keepdim=True)
        hard_neg = torch.mean(sorted_sim[:, -self.bg_nodes:], dim=1, keepdim=True)

        logits = torch.cat([pos, hard_neg], dim=1)

        target = logits.new_zeros(B).long()
        loss = self.criterion(logits, target)
        loss_dict = {'dis': loss}

        indexes = sorted_indexes[:, :self.fg_nodes].unsqueeze(2).unsqueeze(3).expand(B, self.fg_nodes, *av_heatmaps.shape[2:])
        fg_heatmaps = torch.gather(att_heatmaps, 1, indexes)
        if 'div' in self.loss_names:
            combined_heatmaps = torch.mean(fg_heatmaps, dim=1, keepdim=True)
            loss_dict['div'] = get_div_loss(combined_heatmaps, av_heatmaps, self.div_mode)
        return loss_dict
