import torch
from .orepa import OREPA
from .dysample import Dy_Sample


class HANCLayer(torch.nn.Module):

    def __init__(self, in_chnl, out_chnl, k):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC
        """

        super(HANCLayer, self).__init__()

        self.k = k
        self.up1 = Dy_Sample(in_chnl,2)
        self.up2 = Dy_Sample(in_chnl, 4)
        self.up3 = Dy_Sample(in_chnl, 8)
        self.up4 = Dy_Sample(in_chnl, 16)
        self.cnv = OREPA((2 * k - 1) * in_chnl, out_chnl)
        self.act = torch.nn.SiLU(inplace=True)
        self.bn = torch.nn.BatchNorm2d(out_chnl)

    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.concat(
                [
                    x,
                    self.up1(torch.nn.AvgPool2d(2)(x)),
                    self.up1(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=1,
            )

        elif self.k == 3:
            x = torch.concat(
                [
                    x,
                    self.up1(torch.nn.AvgPool2d(2)(x)),
                    self.up2(torch.nn.AvgPool2d(4)(x)),
                    self.up1(torch.nn.MaxPool2d(2)(x)),
                    self.up2(torch.nn.MaxPool2d(4)(x)),
                ],
                dim=1,
            )

        elif self.k == 4:
            x = torch.concat(
                [
                    x,
                    self.up1(torch.nn.AvgPool2d(2)(x)),
                    self.up2(torch.nn.AvgPool2d(4)(x)),
                    self.up3(torch.nn.AvgPool2d(8)(x)),
                    self.up1(torch.nn.MaxPool2d(2)(x)),
                    self.up2(torch.nn.MaxPool2d(4)(x)),
                    self.up3(torch.nn.MaxPool2d(8)(x)),
                ],
                dim=1,
            )

        elif self.k == 5:
            x = torch.concat(
                [
                    x,
                    self.up1(torch.nn.AvgPool2d(2)(x)),
                    self.up2(torch.nn.AvgPool2d(4)(x)),
                    self.up3(torch.nn.AvgPool2d(8)(x)),
                    self.up4(torch.nn.AvgPool2d(16)(x)),
                    self.up1(torch.nn.MaxPool2d(2)(x)),
                    self.up2(torch.nn.MaxPool2d(4)(x)),
                    self.up3(torch.nn.MaxPool2d(8)(x)),
                    self.up4(torch.nn.MaxPool2d(16)(x)),
                ],
                dim=1,
            )

        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x


class HCABlock(torch.nn.Module):

    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'SiLU'.
            k (int, optional): k in HANC. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in HANC. Defaults to 4.
        """

        super().__init__()

        self.conv1 = OREPA(n_filts, n_filts * inv_fctr)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.conv2 = OREPA(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts, k)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = OREPA(n_filts, out_channels)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

        self.activation = torch.nn.SiLU(inplace=True)


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)

        return x