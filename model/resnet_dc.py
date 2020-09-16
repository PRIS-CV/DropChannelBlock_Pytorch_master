import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # ### strategy1: drop with fixed rate ###
    # def drop_channel_block(self, x, drop_rate=0.1):
    #     """
    #     x: bs*c*h*w
    #     """
    #     if self.training:
    #         ### generate correlation matrix
    #         bs, c, h, w = x.size()
    #         mask = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x) #bs*c*h*w
    #         mask = mask.view(bs,c,-1)
    #         loc = torch.argmax(mask, dim=2) #bs*c
    #         loc_x = loc / w
    #         loc_y = loc - loc_x*w
    #         loc_xy = torch.cat((loc_x.unsqueeze(-1), loc_y.unsqueeze(-1)), -1).float()
    #         # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    #         mask_square = torch.sum(loc_xy*loc_xy, -1)
    #         dot_product = torch.bmm(loc_xy, torch.transpose(loc_xy, 1, 2))

    #         distances = mask_square.unsqueeze(2) - 2.0*dot_product + mask_square.unsqueeze(1)
    #         distances = torch.sqrt(distances)

    #         distances = -distances
    #         d_min = torch.min(distances, dim=-1, keepdim=True)[0]
    #         d_max = torch.max(distances, dim=-1, keepdim=True)[0]
    #         adjs = (distances - d_min) / (d_max - d_min)
    #         ### drop function
    #         topk = int(c*drop_rate)
    #         value, indices = torch.topk(adjs, topk, dim = 2) # [bs, c, c] -> [bs, c, topk]
    #         value = value[:,:,-1].view(bs, c, -1)
    #         mask = adjs < value # [bs, c, c]

    #         drop_index = torch.randint(low=0, high=c, size=(bs,1,1)).to(distances.device).repeat(1,1,c).long()
    #         mask = torch.gather(mask, 1, drop_index).view(bs,-1,1,1).repeat(1,1,h,w).float() #[bs,c,h,w]

    #         ### get masked heatmap and remained heatmap for visualization
    #         mask_remain = mask
    #         mask_drop = 1 - mask_remain
    #         x_remain = x* mask_remain
    #         x_drop = x* mask_drop
    #         x_select_channel = torch.gather(x, 1, drop_index[:,:,0].view(bs,1,1,1).repeat(1,1,h,w))
    #         ### normalize
    #         x_out = x_remain / (1-drop_rate)

    #         return x_out, torch.sum(x_remain, dim=1, keepdim=True), torch.sum(x_drop, dim=1, keepdim=True), x_select_channel
    #     else:
    #         return x, torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True)

    # ### strategy2: drop with fixed distance ###
    # def drop_channel_block(self, x, thr_rate=1/6):
    #     """
    #     x: bs*c*h*w
    #     """
    #     if self.training:
    #         ### generate correlation matrix
    #         bs, c, h, w = x.size()
    #         mask = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x) #bs*c*h*w
    #         mask = mask.view(bs,c,-1)
    #         loc = torch.argmax(mask, dim=2) #bs*c
    #         loc_x = loc / w
    #         loc_y = loc - loc_x*w
    #         loc_xy = torch.cat((loc_x.unsqueeze(-1), loc_y.unsqueeze(-1)), -1).float()
    #         # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    #         mask_square = torch.sum(loc_xy*loc_xy, -1)
    #         dot_product = torch.bmm(loc_xy, torch.transpose(loc_xy, 1, 2))

    #         distances = mask_square.unsqueeze(2) - 2.0*dot_product + mask_square.unsqueeze(1)
    #         distances = torch.sqrt(distances)

    #         ### drop function
    #         drop_thr = w*thr_rate
    #         mask = distances > drop_thr

    #         drop_index = torch.randint(low=0, high=c, size=(bs,1,1)).to(distances.device).repeat(1,1,c).long()
    #         mask = torch.gather(mask, 1, drop_index).view(bs,-1,1,1).repeat(1,1,h,w).float() #[bs,c,h,w]

    #         ### get masked heatmap and remained heatmap for visualization
    #         mask_remain = mask
    #         mask_drop = 1 - mask_remain
        
    #         x_remain = x* mask_remain
    #         x_drop = x* mask_drop
    #         x_select_channel = torch.gather(x, 1, drop_index[:,:,0].view(bs,1,1,1).repeat(1,1,h,w))
    #         ### normalize
    #         x_out = x_remain / (torch.sum(mask[:,:,:1,:1], dim=1, keepdim=True) / mask.size(1))

    #         return x_out, torch.sum(x_remain, dim=1, keepdim=True), torch.sum(x_drop, dim=1, keepdim=True), x_select_channel
    #     else:
    #         return x, torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True)

    ### strategy3: drop based on BP ###
    def drop_channel_block(self, x, drop_rate=0.05):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #64
        
        # x = self.drop_channel_block(x) # add dc block v1
        x = self.layer1(x) #256
        # x = self.drop_channel_block(x) # add dc block v2
        x = self.layer2(x) #512 
        x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = self.drop_channel_block(x) # add dc block v3
        x = self.layer3(x) #1024
        # x = self.drop_channel_block(x) # add dc block v4
        x = self.layer4(x) #2048
        # x = self.drop_channel_block(x) # add dc block v5

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel
        # return x, [], [], [], [], []


def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(num_classes, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
