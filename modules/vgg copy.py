import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from modules.layers import *
from util import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
# TODO : Avgpool만 하면 됨.. 그전에 RSP adversarial attack 되는지 해보자.

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True,name=''):
        super(VGG, self).__init__()
        self.name = name
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7),name=name + '.avgpool')
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, num_classes), name=name + '.classifier'
        )
        self.X_pure = None
        if init_weights:
            self._initialize_weights()

    def forward(self, x,inference_pkg):
        x = self.features(x,inference_pkg)
        inference_pkg.output_dict[self.name + '.features'] = x
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x,inference_pkg)
        return x
    
    def get_ideal_feature_vector(self,idx,inference_pkg):

        X = inference_pkg.output_dict['vgg16.classifier.1']
       
        #X_pure = X.clone().detach().requires_grad_()
        X_pure = torch.ones_like(X,requires_grad=True,device=inference_pkg.device)
        y = torch.zeros((1,1000),device=inference_pkg.device)
        _ = InferencePkg()
        _.device = inference_pkg.device
        
        #y[:,idx] = org_out.clone().detach()[:,idx]
        y[:,idx] = 1

        optimizer = torch.optim.Adam([X_pure], lr=0.0005,weight_decay=0.1)

        Loss = torch.nn.CosineEmbeddingLoss()
        if self.X_pure == None:
            for i in range(10000):
                optimizer.zero_grad()

                pure_out = self.classifier[3](X_pure,_)
                pure_out = self.classifier[4](pure_out,_)
                pure_out = self.classifier[5](pure_out,_)
                pure_out = self.classifier[6](pure_out,_)
                pure_out_2 = F.softmax(pure_out)

                # loss1 = F.mse_loss(y, pure_out,reduction='sum')
                # loss2 = F.mse_loss(y[:,idx], pure_out[:,idx])
                # loss = loss1 + 5 * loss2
                loss1 = F.mse_loss(pure_out_2,y)
                loss2 = F.mse_loss(pure_out_2[:,282],y[:,282])
                loss = loss1 + 5 * loss2
                loss.backward()
                optimizer.step()
                if loss < 0.5:
                    break
        
        self.X_pure = X_pure





    def get_feature_contribution(self,out_f,y,classifier,inference_pkg,idx=None):



        






        self.get_ideal_feature_vector(idx,inference_pkg)




        #last_out = y
        last_out = self.X_pure
        _,c_class = y.shape
        n,c,h,w = out_f.shape
        cont = torch.zeros((n,h,w),device=out_f.device)
        stimulation = torch.zeros((n,h,w,c,h,w),device=out_f.device)
        for _h in range(h):
            for _w in range(w):
                stimulation[:,_h,_w,:,_h,_w] = out_f[:,:,_h,_w]
        stimulation = stimulation.view(n,h,w,-1)
        out = classifier[0].decomp_forward(stimulation,inference_pkg,bias_shrink=h*w) #out.shape = (N,H,W,1000). last_out.shape = (N,1000)
        out = classifier[1].decomp_forward(out,inference_pkg,bias_shrink=h*w)
        #out = classifier.forward(stimulation,inference_pkg)
        last_out = last_out.view(n,1,1,4096)
        cos = nn.CosineSimilarity(dim=3, eps=1e-12)
            #calculate simmap for given position
        simmap = cos(out,last_out) * torch.norm(out,dim=3) / (torch.norm(last_out,dim=3) + 1e-12) # the contribution
        return simmap
    def get_pixel_cont(self,input,y,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),idx=None):
        """
        y : 벡터. (N,C) or scalar
        """
        
        o = SimmapObj()
        o.model_out = y
        o.cav_contrastive_rule = contrastive_rule
        out_f= inference_pkg.output_dict[self.name + '.features']
        o.simmap = self.get_feature_contribution(out_f,y,self.classifier,inference_pkg,idx=idx)
        inference_pkg.simmaps['classifier'] = o.simmap.clone().cpu().detach()
        o.simmap = o.cav_contrastive_rule(o.simmap)

        o = self.features.simprop2(o,inference_pkg)
        return o.simmap

    def get_ERF(self,input,extern_simmap,inference_pkg,target_layer_name=None):
        """
        extern_simmap : target layer의 H,W를 가진 (N,H,W) simmap. 
        
        """

        if target_layer_name == None:
            raise 
        else:
            o = SimmapObj()
            o.target_layer_name = target_layer_name
            o.external_simmap =extern_simmap
            o.simmap = torch.zeros((1,7,7),device=inference_pkg.device) # 나중에 수정. feature output size로
            o = self.features.simprop2(o,inference_pkg)
        return o.simmap

    

    def relprop(self, R, alpha):
        x = self.classifier.relprop(R, alpha)
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.relprop(x, alpha)
        x = self.features.relprop(x, alpha)

        return x

    def RSP(self, R):
        x1 = self.features[:-1].RSP_relprop(R)

        return x1
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False,name=''):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2,return_indices=True)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return Sequential(*layers,name=name + '.features')


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'],name='vgg16'),**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        # conv = Conv2d(512,4096,kernel_size=7)
        # conv.bias = model.classifier[0].bias
        # with torch.no_grad():
        #     conv.weight.copy_(model.classifier[0].weight.view((4096,512,7,7))) # 첫번째 linear layer를 동등한 convlayer로 바꿔줌
        # model.classifier[0] = conv
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
