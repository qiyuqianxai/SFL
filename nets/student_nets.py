
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# torch.cuda.set_device(1)
# maybe it can be smaller
class StuNet(nn.Module):
    def __init__(self, model_name="vgg11"):
        super(StuNet, self).__init__()

        # encoding components
        basenet = getattr(models,model_name)(pretrained=True)
        if model_name == "resnet18":
            modules = list(basenet.children())[:-1]  # delete the last fc layer.
            self.encoder = nn.Sequential(*modules)
        else:
            modules = list(basenet.children())[:-1]  # delete the last fc layer.
            self.encoder = nn.Sequential(*modules)
            self.encoder.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # Decoder
        self.decoder = nn.Linear(512,10)
        # self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        # self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        # self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.k4, stride=self.s4,
        #                        padding=self.pd4),
        #     nn.BatchNorm2d(16, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.k3, stride=self.s3,
        #                        padding=self.pd3),
        #     nn.BatchNorm2d(8, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
        #                        padding=self.pd2),
        #     nn.BatchNorm2d(3, momentum=0.01),
        #     nn.Tanh()  # y = (y1, y2, y3) \in [0 ,1]^3
        # )

    def forward(self, x, latent=None):
        if latent==None:
            z = self.encoder(x)
            z = z.view(-1, 512)  # flatten output of conv 512
            return z
        else:
            z = latent.view(-1,512)
            z = F.relu(z)
            x_reconst = self.decoder(z)
            # x_reconst = F.interpolate(x_reconst, size=(64, 64), mode='bilinear', align_corners=True)
            return x_reconst

from torchsummary import summary
if __name__ == '__main__':
    model_name = "vgg11" # resnet18、resnet34、vgg11
    net = StuNet(model_name)
    print(net.encoder)
    summary(net,(3,224,224),batch_size=1,device="cpu")