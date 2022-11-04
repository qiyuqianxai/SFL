import torch
from torch import nn
from nets.pytorch_pretrained_vit import ViT
import torch.nn.functional as F
import torchvision.models as models
class MentorNet(nn.Module):
    def __init__(self,model_name="ViT"):
        super(MentorNet, self).__init__()
        # encoding components
        if model_name == "ViT":
            self.encoder = ViT('B_16',pretrained=False)
            weights = torch.load("nets/pytorch_pretrained_vit/B_16.pth",map_location="cpu")
            self.encoder.load_state_dict(weights)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
        else:
            self.encoder = models.resnet152(pretrained=True)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
            # modules = list(self.encoder.children())[:-1]  # delete the last fc layer.
            # self.encoder = nn.Sequential(*modules)
        # Decoder
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.decoder = nn.Linear(512,10) # do classify
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
            z = z.view(z.size(0), 512)  # flatten output of conv 512
            return z
        else:
            z = latent.view(latent.size(0), 512)
            z = F.relu(z)
            x_reconst = self.decoder(z)
            # x_reconst = F.interpolate(x_reconst, size=(64, 64), mode='bilinear', align_corners=True)
            return x_reconst


from torchsummary import summary
if __name__ == '__main__':
    # net = MaskedAutoencoderViT()
    # # net.to("cuda")
    # checkpoint = torch.load("mae_visualize_vit_large_ganloss.pth", map_location='cpu')
    # net.load_state_dict(checkpoint['model'], strict=False)
    # summary(net, (3, 224, 224),device="cpu")
    model_name = "resnet152"
    model = MentorNet(model_name)
    summary(model, (3, 224, 224),device="cpu")