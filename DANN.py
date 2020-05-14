import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
from torchvision.models import alexnet

class ReverseLayer(Function):
    
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

    
    
class DANN(nn.Module):
    
    
    def __init__(self, num_classes = 1000, pretrained = False):
        
        super(DANN, self).__init__()
        alexnet_model = alexnet(pretrained = pretrained)
        
        self.gf = alexnet_model.features
        self.avgpool = alexnet_model.avgpool
        self.gy = alexnet_model.classifier
        self.gd = alexnet_model.classifier
        
        self.gy[6] = nn.Linear(4096, num_classes)
        self.gd[6] = nn.Linear(4096, 2)
        
        
    def forward(self, x, alpha = None):
        
        x = self.gf(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if alpha is not None:
            x = ReverseLayer.apply(x, alpha)
            x = self.gd(x)
        else:
            x = self.gy(x)
            
        return x


    
def dann(pretrained = False, **kwargs):
    
    model = DANN(pretrained = pretrained, **kwargs)
        
    return model