import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
MEL
'''
class TripletNet(nn.Module):
    def __init__(self, dropout_rate):
        super(TripletNet, self).__init__()
        self.dropout_rate = dropout_rate
       
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d((2,4)),
                nn.Dropout(dropout_rate))
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 384,3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.MaxPool2d((4,5)),
                nn.Dropout(dropout_rate))
        self.conv3 = nn.Sequential(
                nn.Conv2d(384, 768, 3, stride=1, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                nn.MaxPool2d((3,8)),
                nn.Dropout(dropout_rate))
        self.conv4 = nn.Sequential(
                nn.Conv2d(768, 2048, 3, stride=1, padding=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.MaxPool2d((4,8)),
                nn.Dropout(dropout_rate))

        self.dense = nn.Linear(2048,128)
        

    def forward_single(self,x):
        # x.shape = [batch_size, 96, 1366]
        if x.shape[1] != 96 or x.shape[2] != 1366:
            print ("Input shape is not correct (%d, %d)"%(x.shape[1], x.shape[2]))
            return Variable(torch.cuda.FloatTensor(1,128).zero_())
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size()[0], -1)
        out = self.dense(out)
        return out

    def forward(self, pos1, pos2, neg):
        out_pos1 = self.forward_single(pos1)
        out_pos2 = self.forward_single(pos2)
        out_neg = self.forward_single(neg)
        return out_pos1, out_pos2, out_neg

class TripletNet2(nn.Module):
    def __init__(self, dropout_rate):
        super(TripletNet2, self).__init__()
        self.dropout_rate = dropout_rate
       
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d((2,4)),
                nn.Dropout(dropout_rate))
        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 384,3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.MaxPool2d((4,5)),
                nn.Dropout(dropout_rate))
        self.conv3 = nn.Sequential(
                nn.Conv2d(384, 768, 3, stride=1, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(),
                nn.MaxPool2d((3,8)),
                nn.Dropout(dropout_rate))
        self.conv4 = nn.Sequential(
                nn.Conv2d(768, 2048, 3, stride=1, padding=1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.MaxPool2d((4,8)),
                nn.Dropout(dropout_rate))

        self.dense = nn.Linear(2048,256)
        
        #downsample 1, 1/4 to (48, 341)
        # self.avgpool_2_4 = nn.MaxPool2d((2,4))
        
        # 96 , 1366 -(2,4)-> 48,341 -(4,5)> 12,68 -(3,8)-> 4,8 -(4,8)-> 1,1
        self.sampleconv1 = nn.Sequential(
                nn.Conv2d(1,128,8, stride=1, padding=4),
                nn.ReLU(),
                nn.MaxPool2d((2,4)),
                nn.Dropout(dropout_rate))
        self.sampleconv2 = nn.Sequential(
                nn.Conv2d(128,256,8, stride=1, padding=4),
                nn.ReLU(),
                nn.MaxPool2d((4,5)),
                nn.Dropout(dropout_rate))
        
        self.sampleconv3 = nn.Sequential(
                nn.Conv2d(256,512,4, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d((3,8)),
                nn.Dropout(dropout_rate))
        self.sampleconv4 = nn.Sequential(
                nn.Conv2d(512,1024,3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d((4,8)),
                nn.Dropout(dropout_rate))
        # out (1,1,1024)

        self.linear = nn.Linear(1024,256) 


    def forward_single(self,x):
        # x.shape = [batch_size, 96, 1366]
        if x.shape[1] != 96 or x.shape[2] != 1366:
            print ("Input shape is not correct (%d, %d)"%(x.shape[1], x.shape[2]))
            return Variable(torch.cuda.FloatTensor(1,128).zero_())
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        #convnet
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)
        out1 = self.conv4(out1)
        out1 = out1.view(out1.size()[0], -1)
        out1 = self.dense(out1)

        #subsampling
        # out2 = self.avgpool_2_4(x)
        out2 = self.sampleconv1(x)
        out2 = self.sampleconv2(out2)
        out2 = self.sampleconv3(out2)
        out2 = self.sampleconv4(out2)
        out2 = out2.view(out2.size()[0], -1)
        out2 = self.linear(out2)
        #out2 = F.normalize(out2)
        
        out = torch.cat((out1, out2),1)
        return out

    def forward(self, pos1, pos2, neg):
        out_pos1 = self.forward_single(pos1)
        out_pos2 = self.forward_single(pos2)
        out_neg = self.forward_single(neg)
        return out_pos1, out_pos2, out_neg


if __name__ =='__main__':
    model = TripleNet(0.5)
    print (model)
