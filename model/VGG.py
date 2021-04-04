import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['vgg19_bn']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, cdb_flag, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        if cdb_flag == "max_activation":
            self.cdb_metric = self.drop_channel_block_s1
        elif cdb_flag == "bilinear_pooling":
            self.cdb_metric = self.drop_channel_block_s2
        else:
            self.cdb_metric = lambda x: {x, [], [], [], [], []}
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    ### strategy1: cal correlation based on Euclidean distance ###
    def drop_channel_block_s1(self, x, drop_rate=0.2):
        """
        x: bs*c*h*w
        """
        if self.training:
            ### generate correlation matrix
            bs, c, h, w = x.size()
            mask = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x) #bs*c*h*w
            mask = mask.view(bs,c,-1)
            loc = torch.argmax(mask, dim=2) #bs*c
            loc_x = loc / w
            loc_y = loc - loc_x*w
            loc_xy = torch.cat((loc_x.unsqueeze(-1), loc_y.unsqueeze(-1)), -1).float()
            # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
            mask_square = torch.sum(loc_xy*loc_xy, -1)
            dot_product = torch.bmm(loc_xy, torch.transpose(loc_xy, 1, 2))

            distances = mask_square.unsqueeze(2) - 2.0*dot_product + mask_square.unsqueeze(1)
            distances = torch.sqrt(distances)

            distances = -distances
            d_min = torch.min(distances, dim=-1, keepdim=True)[0]
            d_max = torch.max(distances, dim=-1, keepdim=True)[0]
            adjs = (distances - d_min) / (d_max - d_min)
            ### drop function
            topk = int(c*drop_rate)
            value, indices = torch.topk(adjs, topk, dim = 2) # [bs, c, c] -> [bs, c, topk]
            value = value[:,:,-1].view(bs, c, -1)
            mask = adjs < value # [bs, c, c]

            drop_index = torch.randint(low=0, high=c, size=(bs,1,1)).to(distances.device).repeat(1,1,c).long()
            mask = torch.gather(mask, 1, drop_index).view(bs,-1,1,1).repeat(1,1,h,w).float() #[bs,c,h,w]

            ### get masked heatmap and remained heatmap for visualization
            mask_remain = mask
            mask_drop = 1 - mask_remain
            x_remain = x* mask_remain
            x_drop = x* mask_drop
            x_select_channel = torch.gather(x, 1, drop_index[:,:,0].view(bs,1,1,1).repeat(1,1,h,w))
            ### normalize
            x_out = x_remain / (1-drop_rate)

            return x_out, torch.sum(x, dim=1, keepdim=True), torch.sum(x_remain, dim=1, keepdim=True), torch.sum(x_drop, dim=1, keepdim=True), x_select_channel, x[0].unsqueeze(1)
        else:
            return x, torch.sum(x, dim=1, keepdim=True), [], [], [], []
            
    ### strategy2: cal correlation based on BP ###
    def drop_channel_block_s2(self, x, drop_rate=0.05):
        """
        x: bs*c*h*w
        """
        if self.training:
            ### generate correlation matrix
            bs, c, w, h = x.size()
            x_norm = x.view(bs, c, -1)
            x_norm = torch.nn.functional.normalize(x_norm, dim = 2)
            adjs = torch.bmm(x_norm, torch.transpose(x_norm, 1, 2))

            # mask = adjs > drop_thr
            ### drop function
            topk = int(c*drop_rate)
            value, indices = torch.topk(adjs, topk, dim = 2) # [bs, c, c] -> [bs, c, topk]
            value = value[:,:,-1].view(bs, c, -1)
            mask = adjs < value # [bs, c, c]
            
            drop_index = torch.randint(low=0, high=c, size=(bs,1,1)).to(adjs.device).repeat(1,1,c).long()
            mask = torch.gather(mask, 1, drop_index).view(bs,-1,1,1).repeat(1,1,h,w).float() #[bs,c,h,w]

            ### get masked heatmap and remained heatmap for visualization
            mask_remain = mask
            mask_drop = 1 - mask_remain
        
            x_remain = x* mask_remain
            x_drop = x* mask_drop
            x_select_channel = torch.gather(x, 1, drop_index[:,:,0].view(bs,1,1,1).repeat(1,1,h,w))
            ### normalize
            x_out = x_remain / (1-drop_rate)

            return x_out, torch.sum(x, dim=1, keepdim=True), torch.sum(x_remain, dim=1, keepdim=True), torch.sum(x_drop, dim=1, keepdim=True), x_select_channel, x[0].unsqueeze(1)
        else:
            return x, torch.sum(x, dim=1, keepdim=True), [], [], [], []

    def forward(self, x):
        #[3,448,448]
        x = nn.Sequential(*list(self.features.children())[:7])(x) #[64,224,224]
        x = nn.Sequential(*list(self.features.children())[7:14])(x) #[128,112,112]
        x = nn.Sequential(*list(self.features.children())[14:27])(x) #[256,56,56]
        ### add cdb block v3
        x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = self.cdb_metric(x)
        x = nn.Sequential(*list(self.features.children())[27:40])(x) #[512,28,28]
        x = nn.Sequential(*list(self.features.children())[40:])(x) #[512,14,14]

        return x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, cdb_flag, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), cdb_flag, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg19_bn(pretrained=False, cdb_flag="none", progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, cdb_flag, progress, **kwargs)