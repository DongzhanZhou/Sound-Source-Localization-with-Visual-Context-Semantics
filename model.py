import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models import resnet18
from criterion import CELocLoss, AttCELoss

__all__ = ['LocModel', 'BaseLocModel']

class GlueNet(nn.Module):
    def __init__(self, args):
        super(GlueNet, self).__init__()
        self.num_n = args.num_mid
        self.conv_proj = nn.Conv2d(512, args.num_mid, kernel_size=1, bias=False)
        self.dropout1 = nn.Dropout(args.dropout1)
        self.dropout2 = nn.Dropout(args.dropout2)
    def forward(self, img_feat):
        B,C,h,w = img_feat.shape
        x_proj = self.dropout1(self.conv_proj(img_feat)) # [B, num_node, h, w]
        x_proj_reshaped = x_proj.view(B, self.num_n, -1) # [B, num_node, h*w]
        x_state_reshaped = self.dropout2(img_feat.view(B, C, -1)) # [B, num_state, h*w]
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        return x_proj, x_n_state

class LocModel(nn.Module):
    def __init__(self, args):
        super(LocModel, self).__init__()
        self.imgnet = resnet18(args=args, modal='vision', pretrained=args.imgnet_pretrained)
        self.audnet = resnet18(args=args, modal='audio')
        self.scale = nn.Parameter(torch.ones(1))
        self.gluenet = GlueNet(args)

        self.audio_pool = {'maxpool': nn.AdaptiveMaxPool2d((1, 1)), \
                           'avgpool': nn.AdaptiveAvgPool2d((1, 1))}[args.audio_pool]
        self.args = args
        self.loc_crit = CELocLoss(args)
        self.dis_crit = AttCELoss(args)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def get_losses(self, A0, A, img_feat, aud_feat):
        att_heatmaps, att_feat = self.gluenet(img_feat)
        loss_dict = self.dis_crit(att_feat, aud_feat, att_heatmaps, A)
        loc_loss, logits = self.loc_crit(A0)
        loss_dict['loc'] = loc_loss
        loss_dict['logits'] = logits
        return loss_dict

    def forward(self, image, audio):
        B = image.shape[0]
        img_feat = self.imgnet(image)
        aud_feat = self.audnet(audio)
        aud_feat = self.audio_pool(aud_feat).view(B, -1)

        norm_img_feat = F.normalize(img_feat, dim=1)
        norm_aud_feat = F.normalize(aud_feat, dim=1)
        A0 = torch.einsum('bchw,nc->bnhw', [norm_img_feat, norm_aud_feat]) * self.scale
        A = torch.einsum('bchw,bc->bhw', [norm_img_feat, norm_aud_feat]).unsqueeze(1) * self.scale

        if self.training:
            return self.get_losses(A0, A, img_feat, norm_aud_feat)
        else:
            att_heatmap, _ = self.gluenet(img_feat)
            return A, att_heatmap

class BaseLocModel(nn.Module):
    def __init__(self, args):
        super(BaseLocModel, self).__init__()
        self.imgnet = resnet18(args=args, modal='vision', pretrained=args.imgnet_pretrained)
        self.audnet = resnet18(args=args, modal='audio')
        self.scale = nn.Parameter(torch.ones(1))

        self.audio_pool = {'maxpool': nn.AdaptiveMaxPool2d((1, 1)), \
                           'avgpool': nn.AdaptiveAvgPool2d((1, 1))}[args.audio_pool]
        self.args = args
        self.loc_crit = CELocLoss(args)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, audio, aug_image=None):
        B = image.shape[0]
        img_feat = self.imgnet(image)
        aud_feat = self.audnet(audio)
        aud_feat = self.audio_pool(aud_feat).view(B, -1)

        norm_img_feat = F.normalize(img_feat, dim=1)
        norm_aud_feat = F.normalize(aud_feat, dim=1)
        A0 = torch.einsum('bchw,nc->bnhw', [norm_img_feat, norm_aud_feat]) * self.scale
        A = torch.einsum('bchw,bc->bhw', [norm_img_feat, norm_aud_feat]).unsqueeze(1) * self.scale

        if self.training:
            loss, logits = self.loc_crit(A0)
            return {'loc': loss, 'logits': logits}
        else:
            return A
