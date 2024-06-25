import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from modules.layers import *
from util import *
from collections import defaultdict
from tqdm import tqdm
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model = torchvision.models.vgg16(pretrained=True)

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
        self.ideal_feature_vector_maxpool = defaultdict(lambda : None)
        self.ideal_feature_vector_linear = defaultdict(lambda : None)
        if init_weights:
            self._initialize_weights()
    
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
    def forward(self, x,inference_pkg):
        x = self.features(x,inference_pkg)
        inference_pkg.output_dict[self.name + '.features'] = x
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x,inference_pkg)
        return x
    
    def decomp_forward(self,x,inference_pkg,bias_shrink=1):
        x = self.features.decomp_forward(x,inference_pkg,bias_shrink = bias_shrink)
        x = x.view(x.size(0),-1)
        x = self.classifier.decomp_forward(x,inference_pkg,bias_shrink= bias_shrink)
        return x


    def get_feature_contribution_true(self,out_f,target_out,inference_pkg):
        """
        feature[29] relu (14,14) point-wise feature vector가 원하는 클래스에 대한 각각의 기여 구하는 거
        """
        maxpool_final = self.features[-1]
        linear = self.classifier[0]
        relu = self.classifier[1]
        cos = nn.CosineSimilarity(dim=3, eps=1e-12)
        n,c_o,h_o,w_o = out_f.shape
        total_simmap = torch.zeros((n,h_o,w_o),device=inference_pkg.device)
        last_out = target_out
        
        h_mp = self.features[-1].kernel_size
        w_mp = self.features[-1].kernel_size
        for _h_mp in range(h_mp):
            for _w_mp in range(w_mp):
                stimulus_maxpool = torch.zeros_like(out_f,device=inference_pkg.device)
                stimulus_maxpool[:,:,_h_mp::h_mp,_w_mp::w_mp] = out_f[:,:,_h_mp::h_mp,_w_mp::w_mp]

                out_maxpool = maxpool_final.decomp_forward(stimulus_maxpool,inference_pkg)
                n,c,h,w = out_maxpool.shape
                stimulus_linear = torch.zeros((n,h,w,c,h,w),device = inference_pkg.device)
                for _h in range(h):
                    for _w in range(w):
                        stimulus_linear[:,_h,_w,:,_h,_w] = out_maxpool[:,:,_h,_w]
                stimulus_linear = stimulus_linear.view(n,h,w,-1)
                out = linear.decomp_forward(stimulus_linear,inference_pkg,bias_shrink=h_mp*w_mp*h*w) #out.shape = (N,H,W,1000). last_out.shape = (N,1000)
                #out = relu.decomp_forward(out,inference_pkg,bias_shrink=h_mp*w_mp*h*w)

                last_out = last_out.view(n,1,1,4096)
                
                    #calculate simmap for given position
                simmap = cos(out,last_out) * torch.norm(out,dim=3) / (torch.norm(last_out,dim=3) + 1e-12)
                total_simmap[:,_h_mp::h_mp,_w_mp::w_mp] = simmap
        
        return total_simmap

    def get_feature_contribution_true_v2(self,out_f,target_out,inference_pkg):
        """
        feature[29] relu (14,14) point-wise feature vector가 원하는 클래스에 대한 각각의 기여 구하는 거
        """
        maxpool_final = self.features[-1]
        linear = self.classifier[0]
        relu = self.classifier[1]
        cos = nn.CosineSimilarity(dim=1, eps=1e-12)
        n,c_o,h_o,w_o = out_f.shape
        total_simmap = torch.zeros((n,h_o,w_o),device=inference_pkg.device)
        last_out = target_out
        
        h_mp = self.features[-1].kernel_size
        w_mp = self.features[-1].kernel_size
        last_out = last_out.view(n,4096)
        for _h_mp in range(h_mp):
            for _w_mp in range(w_mp):
                #maxpool 먼저
                stimulus_maxpool = torch.zeros_like(out_f,device=inference_pkg.device)
                stimulus_maxpool[:,:,_h_mp::h_mp,_w_mp::w_mp] = out_f[:,:,_h_mp::h_mp,_w_mp::w_mp]

                out_maxpool = maxpool_final.decomp_forward(stimulus_maxpool,inference_pkg)
                n,c,h,w = out_maxpool.shape
                
                for _h in range(h):
                    for _w in range(w):
                        stimulus_linear = out_maxpool[:,:,_h,_w]
                        lin_weight = linear.weight.view((4096,c,h,w))[:,:,_h,_w]
                        lin_bias = linear.bias
                        out = F.linear(stimulus_linear,lin_weight,lin_bias / (h_mp*w_mp*h*w))
                        simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12)
                        simmap = simmap.view(n,1,1)
                        simmap = bed_of_nail_upsample2d(simmap,(7,7),(_h,_w))
                        simmap = bed_of_nail_upsample2d(simmap,(h_mp,w_mp),(_h_mp,_w_mp))
                        total_simmap += simmap
             
        return total_simmap



        


    def get_feature_contribution_simple(self,out_f,inference_pkg,tau=10,idx=None):

        
        #last_out = self.X_pure
        c_class = 1000
        n,c,h,w = out_f.shape
        device = out_f.device
        t = torch.zeros(n,c_class,h,w,device=device)
        _ = InferencePkg()
        _.device = device
        last_out = torch.zeros((1,1000),device=device)
        last_out[0,idx] = 1
        for _h in range(h):
            for _w in range(w):
                out = torch.zeros((n,c,h,w)).to(device)
                out[:,:,_h,_w] = out_f[:,:,_h,_w]
                out_feat = self.features[-1].decomp_forward(out,inference_pkg)
                out_feat = out_feat.view(n,-1)
                out_feat = self.classifier(out_feat,_)
                #t[:,:,_h,_w] = out_feat
                t[:,:,_h,_w] = out_feat / (out_feat.norm())
        # t_softmax = F.softmax(t*tau,dim=1)
        # simmap = t_softmax[:,idx,:,:]
        simmap = t[:,idx,:,:]
        return simmap
    def get_pixel_cont_simple(self,input,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),idx=None,per_channel=False):
        if inference_pkg.remember_input_graph == True:
            return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=idx,per_channel=per_channel)
        else:
            with torch.no_grad():
                return self._get_pixel_cont_simple(input,inference_pkg,contrastive_rule = contrastive_rule,idx=idx,per_channel=per_channel)

    def _get_pixel_cont_simple(self,input,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),idx=None,per_channel=False):
        """
        if idx == none, 그럼 자기 자신의 feature 벡터 사용.
        """
        

       

        
        o = SimmapObj()
        o.cav_contrastive_rule = contrastive_rule
        #vgg16
        out_f= inference_pkg.output_dict[self.features[-2].name].to(inference_pkg.device)
        #vgg16_bn
        #out_f= inference_pkg.output_dict[self.name + '.features.42'].to(inference_pkg.device)
        #self.get_ideal_feature_vector(idx,inference_pkg)
        last_out = inference_pkg.output_dict[self.classifier[0].name].clone().detach().to(inference_pkg.device)
        if idx == None:
            
            o.simmap = self.get_feature_contribution_true(out_f,last_out,inference_pkg)
        else:
            idx = int(idx)
            #simmap_true = self.get_feature_contribution_v3(out_f,last_out,inference_pkg)
            #simmap_contrast = self.get_feature_contribution_simple(out_f,inference_pkg,tau=tau,idx=idx)
            _input= input.clone().detach()
            _input.requires_grad = True
            #_ = model(_input)
            model_cam = model.to(inference_pkg.device)
            
            with torch.autograd.set_grad_enabled(True):
                target_layers = [model_cam.features[29]]
                cam_targets = [ClassifierOutputTarget(idx)]
                cam = ScoreCAM(model_cam,target_layers,use_cuda=True)
                cam_image = cam(_input,cam_targets)
                expl_img = torch.tensor(cam_image).to(inference_pkg.device)
                simmap_contrast = contrastive_rule(expl_img)
            #o.simmap = simmap_true * simmap_contrast
            o.simmap = simmap_contrast

        inference_pkg.simmaps[self.features[-2].name] = o.simmap.clone().cpu().detach()
        o.simmap = o.cav_contrastive_rule(o.simmap)

        #simprop2 start from features.29
        layer_list = []
        for i in range(len(self.features)):
            if i < len(self.features)-1: # 마지막 maxpool 제외
                layer_list.append(self.features[i])
        features_dummy = Sequential(*layer_list,name=self.name + '.features')

        o = features_dummy.simprop2(o,inference_pkg)
        if per_channel:
            simmap = cal_simmap_input_cr_per_neuron(input,o,inference_pkg)
        else:
            simmap = cal_simmap_cr(input,o,inference_pkg)


        return simmap
       

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
            o.simmap = torch.zeros((1,14,14),device=inference_pkg.device) # 나중에 수정. feature output size로
            layer_list = []
        for i in range(len(self.features)):
            if i < len(self.features)-1: # 마지막 maxpool 제외
                layer_list.append(self.features[i])
        features_dummy = Sequential(*layer_list,name=self.name + '.features')

        o = features_dummy.simprop2(o,inference_pkg)
        simmap = cal_simmap_cr(input,o,inference_pkg)
        return simmap

    


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
    model = VGG(make_layers(cfg['B'], batch_norm=True),name=name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False,name='vgg16', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'],name=name),name=name,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

    return model


def vgg16_bn(pretrained=False,name='vgg16_bn', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True,name=name),name=name, **kwargs)
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
