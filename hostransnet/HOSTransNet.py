from .block import *
from .Config import get_Mynet_config
from thop import profile


class HOSTransNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 64
        block = Resblock
        # self.pool = nn.MaxPool2d(2, 2)
        self.inc = CBN(n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 3,3,1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 3,3,1)  # 128  256
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 2,3,1)  # 256  512
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1,3,1)  # 512  512
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, k,inv_fctr,num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels,k,inv_fctr))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)  # 64 224 224
        x2 = self.down_encoder1(x1)  # 128 112 112
        x3 = self.down_encoder2(x2) # 256 56  56
        x4 = self.down_encoder3(x3)  # 512 28  28
        d5 = self.down_encoder4(x4)  # 512 14  14
        #  CCT
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        #  CCT
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        #  Feature fusion
        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))
        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    config_vit = get_Mynet_config()
    model = HOSTransNet(config_vit, n_channels=3, n_classes=1, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(1, 3, 224, 224)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
