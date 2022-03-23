import torchvision.models as torchmodels
import torch.nn as nn

# class ResNet3D(nn.Module):
#     def __init__(self, pretrained = True):
#         super(ResNet3D, self).__init__()
#         self.pretrained = pretrained
#         self.model = torchmodels.video.r3d_18(pretrained=pretrained)
        
#     def forward(self, x):
#         # take a transpose for model input
#         # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W
#         return self.model(x)


class ResNet3D(nn.Module):
    def __init__(self, pretrained = True, n_classes = 400):
        super(ResNet3D, self).__init__()
        self.pretrained = pretrained
        self.model = torchmodels.video.r3d_18(pretrained=pretrained)
        if(n_classes!=400):
            self.model.fc = nn.Linear(in_features=512, out_features=n_classes)
        self.n_classes = n_classes
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # take a transpose for model input
        # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W

        #if self.n_classes == 400:
        #    return self.sm(self.model(x))
        #return self.sm(self.linear(self.model(x)))
        return self.model(x)
