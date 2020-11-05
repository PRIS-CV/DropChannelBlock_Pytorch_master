import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, num_classes, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.cls = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(512, num_classes)
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
        # x = self.drop_channel_block(x) # add dc block v1
        x = nn.Sequential(*list(self.features.children())[7:14])(x) #[128,112,112]
        x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = self.drop_channel_block_s1(x) # add dc block v2
        x = nn.Sequential(*list(self.features.children())[14:27])(x) #[256,56,56]
        x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = self.drop_channel_block_s1(x) # add dc block v3
        x = nn.Sequential(*list(self.features.children())[27:40])(x) #[512,28,28]
        # x = self.drop_channel_block(x) # add dc block v4
        x = nn.Sequential(*list(self.features.children())[40:])(x) #[512,14,14]
        # x = self.drop_channel_block(x) # add dc block v5

        x = self.avgpool(x) #[512,7,7]
        x = torch.flatten(x, 1)
        x = self.cls(x)
        # return x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel
        return x, [], [], [], [], []

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


def vgg16(num_classes):
    """VGG 16-layer model (configuration "D") with batch normalization

    """
    model = VGG(num_classes, make_layers(cfgs['D'], batch_norm=True))

    return model

def vgg19(num_classes):

    model = VGG(num_classes, make_layers(cfgs['E'], batch_norm=True))

    return model