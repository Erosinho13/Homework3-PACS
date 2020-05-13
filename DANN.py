import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['dann', 'DANN']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayer(Function):
    
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha

    
    
class DANN(nn.Module):
    
    
    def __init__(self, num_classes = 1000):
        
        super(DANN, self).__init__()
        
        self.gf = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.gy = nn.Sequential(
            
            nn.Dropout(),
            
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes)
            
        ) 
        
        self.gd = nn.Sequential(
            
            nn.Dropout(),
            
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 2)
            
        )

        
    def forward(self, x, alpha = None):
        
        x = self.gf(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if alpha is not None:
            x = ReverseLayerF.apply(x, alpha)
            x = self.gd(x)
        else:
            x = self.gy(x)
            
        return x


    
def dann(pretrained = False, progress = True, **kwargs):
    
    model = DANN(**kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress = progress)
        model.load_state_dict(state_dict, strict = False)
        
    return model