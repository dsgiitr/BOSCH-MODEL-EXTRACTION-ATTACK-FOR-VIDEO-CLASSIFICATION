from movinets import MoViNet
from movinets.config import _C


class MoviNet(nn.Module):
    def __init__(self, pretrained = True, n_classes = 400):
        super(MoviNet, self).__init__()
        self.pretrained = pretrained
        self.model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = pretrained)
        #self.linear = nn.Linear(in_features=400, out_features=n_classes)
        self.n_classes = n_classes
        # self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x, print_outputs=False):
        # take a transpose for model input
        # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W
        #if self.n_classes == 600:
        #    return self.sm(self.model(x))
        #return self.linear(self.model(x))
        return self.model(x)
