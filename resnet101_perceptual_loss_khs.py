import torch
import torchvision
import torch.nn as nn

class ResNetPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(ResNetPerceptualLoss, self).__init__()
        blocks = []
        resnet = torchvision.models.resnet101(pretrained=True)
        loss_net = nn.Sequential(*list(resnet.children())[:-2])
        # blocks.append(loss_net.eval())
        for ln in loss_net:
            for p in ln.parameters():
                p.requires_grad = False
        # self.blocks = torch.nn.ModuleList(blocks)
        self.loss_net = loss_net.eval()
        

    def forward(self, input, target):
        x = input
        y = target
        loss = 0.0
        
        x = self.loss_net(x)
        y = self.loss_net(y)
        loss_fn = torch.nn.MSELoss(reduction= 'mean')
        loss = loss_fn(x, y)
        return loss