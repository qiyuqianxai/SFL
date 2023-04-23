from torchsummary import summary
import torch
from torch import nn
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.student_nets import StuNet
from nets.sc_model import ISCNet

class base_net(nn.Module):
    def __init__(self, isc_model, channel_model):
        super(base_net, self).__init__()
        self.isc_model = isc_model
        self.ch_model = channel_model

    def forward(self,x):
        encoding = self.isc_model(x)
        encoding_with_noise = self.ch_model(encoding)
        decoding = self.isc_model(x, encoding_with_noise)
        return decoding

if __name__ == '__main__':
    stu_model = ISCNet("vgg11")
    # mentor_model = ISCNet("resnet152")
    channel_model = channel_net()

    # tst_model = base_net(mentor_model,channel_model)
    # summary(tst_model,(3,224,224),device="cpu")

    model = base_net(stu_model, channel_model)
    # summary(tst_model,(3,224,224),device="cuda")
    checkpoint = torch.load(r"D:\PyProj\FL-LISC\checkpoints\0\student_vgg11_10_rali.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    # summary(model, (3, 224, 224),device="cpu")
    from torchvision.io.image import read_image
    from torchvision.transforms.functional import normalize, resize, to_pil_image
    from torchcam.methods import SmoothGradCAMpp

    cam_extractor = SmoothGradCAMpp(model)
    # Get your input
    img = read_image(r"D:\PyProj\FL-LISC\res2.jpg")
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    decoding = model(input_tensor.unsqueeze(0))
    print(decoding.squeeze(0).argmax().item())
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(decoding.squeeze(0).argmax().item(), decoding)

    import matplotlib.pyplot as plt

    # Visualize the raw CAM
    plt.imshow(activation_map[0].squeeze(0).numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
    from torchcam.utils import overlay_mask

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.show()