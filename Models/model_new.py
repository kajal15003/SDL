import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x

#--------------------------------
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlockO(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlockO, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlockB(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlockB, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlockB(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlockB, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x

#----------------------------------------------------


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       

# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.visible.layer4(x)
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
            
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        x = self.thermal.layer3(x)
        x = self.thermal.layer4(x)
        x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x
        
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 2048
            print(self.thermal_net)


        self.feature = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature1 = FeatureBlockO(pool_dim, low_dim, dropout = drop)
        self.feature2 = FeatureBlockB(pool_dim, low_dim, dropout = drop)
        self.feature3 = FeatureBlockB(pool_dim, low_dim, dropout = drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.l2norm = Normalize(2)
        
    def forward(self, x1, x2, modal = 0):
        if modal==0:
            x1 = self.visible_net(x1)
            x2 = self.thermal_net(x2)
            x = torch.cat((x1,x2), 0) 
            #x=torch.mean(x1,x2)
        elif modal ==1:
            x = self.visible_net(x1)
        elif modal ==2:
            x = self.thermal_net(x2)
        
        #z=torch.zeros(1024)
        #z = Variable(z.cuda())
        #x1=torch.cat((x(1024),z),0)
        #x1n=x1
        #x2n=x2
        #y=x
        #xv = torch.cat((x1,x2), 0)
        #x_inf = torch.cat((x1,x2), 0)
        y = self.feature(x)
        y1 = self.feature1(x)
        y2 = self.feature2(x)
        y3 = self.feature3(x)
        #print(y.size(),y1.size(),y2.size(),y3.size())  
        out = self.classifier(y)
        #print(out.size())
        out1 = self.classifier1(y3)
        #zp=torch.zeros(32,512)
        #zp = Variable(zp.cuda())
        #feat_viszp=torch.cat((y1,zp),0)
        #feat_infzp=torch.cat((zp,y2),0)
        out2=self.classifier1(y1)
        out3=self.classifier1(y2)       
        #y3=y3.t()
        #y4 = torch.add(y1,y3)
        #y5 = torch.add(y2,y3)
        #print(y4.size(),y5.size())
        if self.training:
            return out, self.l2norm(y), self.l2norm(y1), self.l2norm(y2), out1, self.l2norm(y3),out2, out3
        else:
            #return self.l2norm(x), self.l2norm(y3)
            return self.l2norm(x), self.l2norm(y3)#, self.l2norm(y3),self.l2norm(y2), self.l2norm(y3)
            #return self.l2norm(y3)
            
# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)
