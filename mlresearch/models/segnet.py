
import torch.nn as nn
from torch.nn import functional


class SegNet(nn.Module):
    """
    https://ai-pool.com/m/segnet-1567616679

    """
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        def sequential(channels, shift=None):
            layers = []

            for channel1, channel2, in zip(channels, channels[1:]):
                layers += [
                    nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(channel2),
                    nn.ReLU()
                ]

            if shift is not None:
                layers = layers[:shift]

            return nn.Sequential(*layers)

        # encoder (down sampling)
        self.enc_conv0 = sequential([input_channels, 64, 64])
        self.enc_conv1 = sequential([64, 128, 128])
        self.enc_conv2 = sequential([128, 256, 256, 256])
        self.enc_conv3 = sequential([256, 512, 512, 512])
        self.enc_conv4 = sequential([512, 512, 512, 512])

        # decoder (up sampling)
        self.dec_conv4 = sequential([512, 512, 512, 512])
        self.dec_conv3 = sequential([512, 512, 512, 256])
        self.dec_conv2 = sequential([256, 256, 256, 128])
        self.dec_conv1 = sequential([128, 128, 64])
        self.dec_conv0 = sequential([64, 64, output_channels], shift=-2)

    def forward(self, x):

        # encoder (down sampling)
        x1 = self.enc_conv0(x)
        x1p, id1 = functional.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)

        x2 = self.enc_conv1(x1p)
        x2p, id2 = functional.max_pool2d(x2, kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        x3 = self.enc_conv2(x2p)
        x3p, id3 = functional.max_pool2d(x3, kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        x4 = self.enc_conv3(x3p)
        x4p, id4 = functional.max_pool2d(x4, kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        x5 = self.enc_conv4(x4p)
        x5p, id5 = functional.max_pool2d(x5, kernel_size=(2, 2), stride=(2, 2), return_indices=True)

        # decoder (up sampling)
        x5d = functional.max_unpool2d(x5p, id5, kernel_size=(2, 2), stride=(2, 2))
        x51d = self.dec_conv4(x5d)

        x4d = functional.max_unpool2d(x51d, id4, kernel_size=(2, 2), stride=(2, 2))
        x41d = self.dec_conv3(x4d)

        x3d = functional.max_unpool2d(x41d, id3, kernel_size=(2, 2), stride=(2, 2))
        x31d = self.dec_conv2(x3d)

        x2d = functional.max_unpool2d(x31d, id2, kernel_size=(2, 2), stride=(2, 2))
        x21d = self.dec_conv1(x2d)

        x1d = functional.max_unpool2d(x21d, id1, kernel_size=(2, 2), stride=(2, 2))
        x11d = self.dec_conv0(x1d)

        return x11d
