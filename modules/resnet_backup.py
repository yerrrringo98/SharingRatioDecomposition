import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modules.layers import *
from util import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', #org
    #'resnet50' : 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',# v2
    #'resnet50': "https://download.pytorch.org/models/resnet50-0676ba61.pth", # V1
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,name=''):
    """3x3 convolution with padding"""
    name = name
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,name=name)


def conv1x1(in_planes, out_planes, stride=1,name=''):
    """1x1 convolution"""
    name = name
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,name=name)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, name = ''):
        super(BasicBlock, self).__init__()
        self.name = name
        

        self.conv1 = conv3x3(inplanes, planes, stride,name = self.name + '.conv1')
        self.bn1 = BatchNorm2d(planes, name = self.name + '.bn1')
        self.conv2 = conv3x3(planes, planes, name= self.name + '.conv2')
        self.bn2 = BatchNorm2d(planes, name= self.name + '.bn2')
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True,name= self.name + '.relu1')
        self.relu2 = ReLU(inplace=True,name= self.name + '.relu2')

        self.add = Add()


    def forward(self, x,inference_pkg):
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = x.clone().detach().cpu() 
        x1 = x
        x2 = x.clone()

        out = self.conv1(x1,inference_pkg)
        out = self.bn1(out,inference_pkg)
        out = self.relu1(out,inference_pkg)

        out = self.conv2(out,inference_pkg)
        out = self.bn2(out,inference_pkg)

        if self.downsample is not None:
            x2 = self.downsample(x2,inference_pkg)

        out = self.add.forward([out, x2],inference_pkg)
        out = self.relu2(out,inference_pkg)

        return out

    def decomp_forward(self, x,inference_pkg,bias_shrink=1):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1.decomp_forward(x,inference_pkg,bias_shrink=bias_shrink)
        out = self.bn1.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.relu1.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        out = self.conv2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.bn2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        

        if self.downsample is not None:
            x = self.downsample.decomp_forward(x,inference_pkg,bias_shrink=bias_shrink)

        # out = self.add([out, x2])
        out = self.add.decomp_forward([out, x],inference_pkg,bias_shrink=bias_shrink)
        out = self.relu2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        return out
    
    def gamma_forward(self, X,inference_pkg):
        x1 = X
        x2 = X.clone()

        out = self.conv1(x1,inference_pkg)
        out = self.bn1(out,inference_pkg)
        out = self.relu1(out,inference_pkg)

        out = self.conv2(out,inference_pkg)
        out = self.bn2(out,inference_pkg)

        if self.downsample is not None:
            x2 = self.downsample(x2,inference_pkg)

        out = self.add.forward([out, x2],inference_pkg)
        out = self.relu2(out,inference_pkg)

        return out

    def simprop2(self,o,inference_pkg):
        
        o = self.relu2.simprop2(o,inference_pkg)
        o1,o2 = self.add.simprop2(o,inference_pkg)

        if self.downsample is not None:
            o2 = self.downsample.simprop2(o2,inference_pkg)
        
        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        
        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o1 = o1.simprop2_input(block_input,o1,inference_pkg)
        o2 = o2.simprop2_input(block_input,o2,inference_pkg)

        o1.simmap = o1.simmap + o2.simmap
        inference_pkg.simmaps[self.name] = o1.simmap.clone().detach().cpu()
        return o1

    
    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, name = ''):
        super(Bottleneck, self).__init__()
        self.name = name

        self.conv1 = conv1x1(inplanes, planes,name= self.name + '.conv1')
        self.bn1 = BatchNorm2d(planes,name= self.name + '.bn1')
        self.conv2 = conv3x3(planes, planes, stride,name= self.name + '.conv2')
        self.bn2 = BatchNorm2d(planes,name= self.name + '.bn2')
        self.conv3 = conv1x1(planes, planes * self.expansion,name= self.name + '.conv3')
        self.bn3 = BatchNorm2d(planes * self.expansion,name= self.name + '.bn3')
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=True,name= self.name + '.relu1')
        self.relu2 = ReLU(inplace=True,name= self.name + '.relu2')
        self.relu3 = ReLU(inplace=True,name= self.name + '.relu3')

        self.add = Add()


    def forward(self, x,inference_pkg):
        # x1, x2 = self.clone(x, 2)
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = x.clone().detach().cpu() 
        out = self.conv1(x,inference_pkg)
        out = self.bn1(out,inference_pkg)
        out = self.relu1(out,inference_pkg)

        out = self.conv2(out,inference_pkg)
        out = self.bn2(out,inference_pkg)
        out = self.relu2(out,inference_pkg)

        out = self.conv3(out,inference_pkg)
        out = self.bn3(out,inference_pkg)

        if self.downsample is not None:
            x = self.downsample(x,inference_pkg)

        # out = self.add([out, x2])
        out = self.add.forward([out, x],inference_pkg)
        out = self.relu3(out,inference_pkg)

        return out
    
    def decomp_forward(self, x,inference_pkg,bias_shrink=1):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1.decomp_forward(x,inference_pkg,bias_shrink=bias_shrink)
        out = self.bn1.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.relu1.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        out = self.conv2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.bn2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.relu2.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        out = self.conv3.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)
        out = self.bn3.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        if self.downsample is not None:
            x = self.downsample.decomp_forward(x,inference_pkg,bias_shrink=bias_shrink)

        # out = self.add([out, x2])
        out = self.add.decomp_forward([out, x],inference_pkg,bias_shrink=bias_shrink)
        out = self.relu3.decomp_forward(out,inference_pkg,bias_shrink=bias_shrink)

        return out
    
    def gamma_forward(self, X,inference_pkg):
        X2 = X
        X1 = self.conv1.gamma_forward(X,inference_pkg) 
        X1 = self.bn1.gamma_forward(X1,inference_pkg)
        X1 = self.relu1.gamma_forward(X1,inference_pkg)

        X1 = self.conv2.gamma_forward(X1,inference_pkg)
        X1 = self.bn2.gamma_forward(X1,inference_pkg)
        X1 = self.relu2.gamma_forward(X1,inference_pkg)

        X1 = self.conv3.gamma_forward(X1,inference_pkg)
        X1 = self.bn3.gamma_forward(X1,inference_pkg)

        if self.downsample is not None:
            X2 = self.downsample.gamma_forward(X2,inference_pkg)

        # out = self.add([out, x2])
        X = self.add.gamma_forward([X1, X2],inference_pkg)
        X = self.relu3.gamma_forward(X,inference_pkg)

        return X

    
    
    def simprop(self, simmap, inference_pkg):

        """
        map_h,map_w마다 각각의 result map을 만들어주기.
        
        """
        map_h = simmap.shape[0]
        map_w = simmap.shape[1]
        layer_num = int(self.name[-3])
        if layer_num == 2:
            last_block_num = 2
        if layer_num == 3:
            last_block_num = 3
        if layer_num == 4:
            last_block_num = 5
        block_num = int(self.name[-1])
        result_h = self.stride * map_h
        result_w = self.stride * map_w
        result_map = torch.zeros((map_h,map_w,result_h,result_w))
        
        for h in range(map_h):
            for w in range(map_w):
                res_h = self.stride * h
                res_w = self.stride * w
                
                if self.name == 'resnet50.layer1.0':
                    name = self.name[:-2] + '.downsample.1'
                    x = inference_pkg.output_dict[name][0,:,h,w]
                elif self.stride == 2:
                    name = self.name[:-2] + '.downsample.1'
                    x = inference_pkg.output_dict[name][0,:,h,w]


                elif self.stride == 1:
                    name = self.name[:-1] + str(block_num -1) +'.relu3'
                    x = inference_pkg.output_dict[name][0,:,res_h,res_w]
                

                result_vector = inference_pkg.output_dict[self.name + '.relu3'][0,:,h,w]
                por_x = torch.dot(x,result_vector) / torch.norm(result_vector) ** 2
                result_vector_relu2 = inference_pkg.output_dict[self.name + '.relu2'][0,:,h,w]
                debug = torch.zeros_like(result_vector_relu2)
                debug2 = torch.zeros_like(result_vector_relu2)
                debug_conv2 = inference_pkg.output_dict[self.name + '.conv2'][0,:,h,w]
                debug_bn2 = inference_pkg.output_dict[self.name + '.bn2'][0,:,h,w]

                por_v = 1 - por_x
                for i in (-1,0,1):
                    for j in (-1,0,1):
                        if res_h + j < 0 or res_w + i < 0 or res_h + j >= result_h or res_w + i>= result_w:
                            continue

                        target_vector = inference_pkg.output_dict[self.name + '.relu1'][0,:,res_h + j,res_w + i]
                        conv_weight = inference_pkg.weight_dict[self.name + '.conv2'][:,:,j+1,i+1]
                        out = conv_weight @ target_vector
                        debug += out
                        out = out * inference_pkg.bn_zoom_dict[self.name + '.bn2'].cpu()
                        # 여기서 bias의 영향을 더할지 말지 결정하자..
                        bias = inference_pkg.bn_true_bias_dict[self.name + '.bn2'].cpu()
                        por_bias = torch.dot(bias,result_vector_relu2) / torch.norm(result_vector_relu2) ** 2
                        debug2 += out
                        out = out * (inference_pkg.output_dict[self.name + '.relu2'][0,:,h,w] > 0)
                        por_ij = torch.dot(out,result_vector_relu2) / torch.norm(result_vector_relu2) ** 2
                        por_ij += por_bias / 9 # 임시.. 
                        por_ij = por_ij * por_v
                        result_map[h,w,res_h+j,res_w + i] += por_ij
                inference_pkg.simmaps[self.name + 'relu2'] = result_map.sum(dim=1).sum(dim=0)
                result_map[h,w,res_h,res_w] += por_x
                result_map[h,w] = simmap[h,w] * result_map[h,w]
        
        res = result_map.sum(dim=1)
        res = res.sum(dim=0)
        inference_pkg.simmaps[self.name] = res
        return res


    def simprop2(self,o,inference_pkg):
        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o = self.relu3.simprop2(o,inference_pkg)
        o1,o2 = self.add.simprop2(o,inference_pkg)

        if self.downsample is not None:
            o2 = self.downsample.simprop2(o2,inference_pkg)
        
        o1 = self.bn3.simprop2(o1,inference_pkg)
        o1 = self.conv3.simprop2(o1,inference_pkg)
        o1 = self.relu2.simprop2(o1,inference_pkg)

        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        o1 = o1.simprop2_input(block_input,o1,inference_pkg)
        o2 = o2.simprop2_input(block_input,o2,inference_pkg)

        o1.simmap = o1.simmap + o2.simmap
        inference_pkg.simmaps[self.name] = o1.simmap.clone().detach().cpu()
        return o1

    def simprop2_layer1(self,o,inference_pkg,maxpool,relu):
        block_input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        o = self.relu3.simprop2(o,inference_pkg)
        o1,o2 = self.add.simprop2(o,inference_pkg)

        if self.downsample is not None:
            o2 = self.downsample.simprop2(o2,inference_pkg)
        
        o1 = self.bn3.simprop2(o1,inference_pkg)
        o1 = self.conv3.simprop2(o1,inference_pkg)
        o1 = self.relu2.simprop2(o1,inference_pkg)

        o1 = self.bn2.simprop2(o1,inference_pkg)
        o1 = self.conv2.simprop2(o1,inference_pkg)
        o1 = self.relu1.simprop2(o1,inference_pkg)

        o1 = self.bn1.simprop2(o1,inference_pkg)
        o1 = self.conv1.simprop2(o1,inference_pkg)
        o1 = maxpool.simprop2(o1,inference_pkg)
        o1 = relu.simprop2(o1,inference_pkg)
        o2 = maxpool.simprop2(o2,inference_pkg)
        o2 = relu.simprop2(o2,inference_pkg)

        o1.simmap = o1.simmap + o2.simmap
        inference_pkg.simmaps[relu.name] = o1.simmap.clone().detach().cpu()
        return o1



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,name=''):
        super(ResNet, self).__init__()
        
        
        self.is_decomposition = False
        self.name = name
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False,name= self.name + '.conv1')
        # self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                     bias=True)
        self.bn1 = BatchNorm2d(64,name= self.name + '.bn1')
        self.relu = ReLU(inplace=True,name= self.name + '.relu')
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True,name= self.name + '.maxpool')
        self.layer1 = self._make_layer(block, 64, layers[0],name= self.name + '.layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,name= self.name + '.layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,name= self.name + '.layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,name= self.name + '.layer4')
        self.avgpool = AdaptiveAvgPool2d((1, 1),name=self.name + '.avgpool')
        #self.avgpool = AvgPool2d((7, 7), stride=1,name=self.name + '.avgpool')
        self.fc = Linear(512 * block.expansion, num_classes,name = self.name + '.fc')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,name=''):
        name = name
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
                name = name + '.downsample'
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,name=name + '.0'))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,name=name + '.'+str(_)))

        return Sequential(*layers,name=name)

    def forward(self, x,inference_pkg):
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict['input'] = x.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict['input'] = x.clone().detach().cpu() 
     
        x = self.conv1(x,inference_pkg)
        x = self.bn1(x,inference_pkg)
        x = self.relu(x,inference_pkg)
        x = self.maxpool(x,inference_pkg)

        x = self.layer1(x,inference_pkg)
        x = self.layer2(x,inference_pkg)
        x = self.layer3(x,inference_pkg)
        x = self.layer4(x,inference_pkg)

        x = self.avgpool(x,inference_pkg)
        x = x.view(x.size(0), -1)
        x = self.fc(x,inference_pkg)

        return x
    
    def gamma_forward(self,x,inference_pkg):
        x_plus = zero_out_minus(x)
        x_minus = zero_out_plus(x)
        x_static_plus = torch.zeros_like(x,dtype=x.dtype,device=inference_pkg.device,requires_grad=True)
        x_static_minus = torch.zeros_like(x,dtype=x.dtype,device=inference_pkg.device,requires_grad=True)
        X = (x_plus,x_minus,x_static_plus,x_static_minus)
        X = self.conv1.gamma_forward(X,inference_pkg)
        X = self.bn1.gamma_forward(X,inference_pkg)
        X = self.relu.gamma_forward(X,inference_pkg)
        X = self.maxpool.gamma_forward(X,inference_pkg)
        X = self.layer1.gamma_forward(X,inference_pkg)
        X = self.layer2.gamma_forward(X,inference_pkg)
        X = self.layer3.gamma_forward(X,inference_pkg)
        X = self.layer4.gamma_forward(X,inference_pkg)
        x = X[0] + X[1] + X[2] + X[3] # no more relu

        x = self.avgpool(x,inference_pkg)
        x = x.view(x.size(0), -1)
        x = self.fc(x,inference_pkg)
        return x
    
    def decomp_forward(self, x,inference_pkg):
        x = self.conv1.decomp_forward(x,inference_pkg)
        x = self.bn1.decomp_forward(x,inference_pkg)
        x = self.relu.decomp_forward(x,inference_pkg)
        x = self.maxpool.decomp_forward(x,inference_pkg)

        x = self.layer1.decomp_forward(x,inference_pkg)
        x = self.layer2.decomp_forward(x,inference_pkg)
        x = self.layer3.decomp_forward(x,inference_pkg)
        x = self.layer4.decomp_forward(x,inference_pkg)

        x = self.avgpool.decomp_forward(x,inference_pkg)
        x = x.view(x.size(0), -1)
        x = self.fc.decomp_forward(x,inference_pkg)

        return x
    
    def get_pixel_cont(self,input,y,inference_pkg,contrastive_rule = lambda x:(x - x.mean()),rule = None):
        """
        y : 벡터. (N,C) or scalar
        """
        
        o = SimmapObj()
        o.model_out = y
        o.cav_contrastive_rule = contrastive_rule
        simmap = self.simprop2(input,o,inference_pkg,rule=rule)
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
            simmap = self.simprop2(input,o,inference_pkg)
        return simmap

    def simprop2(self,input,o,inference_pkg,rule = None):
        if inference_pkg.remember_input_graph:
            o = self.fc.simprop2(o,inference_pkg) #just pass linear layer
            o = self.avgpool.simprop2(o,inference_pkg,rule = rule) #no relu 그냥 avgpool-linear 결과 계산. 특정 
            o = self.layer4.simprop2(o,inference_pkg)
            o = self.layer3.simprop2(o,inference_pkg)
            o = self.layer2.simprop2(o,inference_pkg)
            o = self.layer1.simprop2_layer1(o,inference_pkg,self.maxpool,self.relu)
            # o = self.maxpool.simprop2(o,inference_pkg)
            # o = self.relu.simprop2(o,inference_pkg)
            o = self.bn1.simprop2(o,inference_pkg)
            o = self.conv1.simprop2(o,inference_pkg)
            o = o.simprop2_input(input,o,inference_pkg)
        else:
            with torch.no_grad():
                o = self.fc.simprop2(o,inference_pkg) #just pass linear layer
                o = self.avgpool.simprop2(o,inference_pkg, rule = rule) #no relu 그냥 avgpool-linear 결과 계산. 특정 
                o = self.layer4.simprop2(o,inference_pkg)
                o = self.layer3.simprop2(o,inference_pkg)
                o = self.layer2.simprop2(o,inference_pkg)
                o = self.layer1.simprop2_layer1(o,inference_pkg,self.maxpool,self.relu)
                inference_pkg.simmaps[self.name + 'relu'] = o.simmap
                # o = self.maxpool.simprop2(o,inference_pkg)
                # o = self.relu.simprop2(o,inference_pkg)
                o = self.bn1.simprop2(o,inference_pkg)
                o = self.conv1.simprop2(o,inference_pkg)
                o = o.simprop2_input(input,o,inference_pkg)


        return o.simmap
        
    def simprop(self,x, simmap, inference_pkg):
        simmap = self.layer4.simprop(simmap,inference_pkg)
        simmap = self.layer3.simprop(simmap,inference_pkg)
        simmap = self.layer2.simprop(simmap,inference_pkg)
        simmap = self.layer1.simprop(simmap,inference_pkg)
        simmap = self.maxpool.simprop(simmap,inference_pkg)

        
        
        
        map_h = simmap.shape[0]
        map_w = simmap.shape[1]
        
        result_h = 2 * map_h
        result_w = 2 * map_w
        result_map = torch.zeros((map_h,map_w,result_h,result_w))
        
        
        for h in range(map_h):
            print(h)
            for w in range(map_w):
                res_h = 2 * h
                res_w = 2 * w
               
                result_vector = inference_pkg.output_dict['resnet50.relu'][0,:,h,w]
                for i in (-3,-2,-1,0,1,2,3):
                    for j in (-3,-2,-1,0,1,2,3):
                        if res_h + j < 0 or res_w + i < 0 or res_h + j >= result_h or res_w + i>= result_w:
                            continue

                        target_vector = x[0,:,res_h + j,res_w + i].cpu()
                        conv_weight = inference_pkg.weight_dict['resnet50.conv1'][:,:,j+3,i+3]
                        out = conv_weight @ target_vector
                        
                        out = out * inference_pkg.bn_zoom_dict['resnet50.bn1'].cpu()
                        # 여기서 bias의 영향을 더할지 말지 결정하자..
                        bias = inference_pkg.bn_true_bias_dict['resnet50.bn1'].cpu()
                        por_bias = torch.dot(bias,result_vector) / torch.norm(result_vector) ** 2
                        
                        out = out * (result_vector > 0)
                        por_ij = torch.dot(out,result_vector) / torch.norm(result_vector) ** 2
                        por_ij += por_bias / 49 # 임시.. 
                        result_map[h,w,res_h+j,res_w + i] += por_ij
                result_map[h,w] = simmap[h,w] * result_map[h,w]
        
        res = result_map.sum(dim=1)
        res = res.sum(dim=0)
        
        return res



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
