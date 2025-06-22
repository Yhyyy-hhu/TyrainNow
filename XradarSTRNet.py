# 作者：姚弘祎
# 修改日期：
# 版本


import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import pytorch_ssim
import torch.optim as optim

gpu_id=3
torch.manual_seed(1)
criterion = torch.nn.MSELoss()
def tempodiffFunc(X,Y):
    if len(X.shape) !=4:
        print('shape不对')
    if X.shape != Y.shape:
        print('无法计算')
    BS = X.shape[0]
    C = X.shape[1]
    batchDiff = torch.zeros((BS, 1))
    batchDiff = batchDiff.cuda(gpu_id)
    for i in range(BS):
        tempoDiff = torch.zeros((19, 1))
        tempoDiff = tempoDiff.cuda(gpu_id)
        for j in range(1,C):
            pred_tempodiff = X[i, j, :, :] - X[i, j - 1, :, :]
            label_tempodiff = Y[i, j, :, :] - Y[i, j - 1, :, :]
            tempoDiff[j - 1] = criterion(pred_tempodiff, label_tempodiff)
        batchDiff[i] = torch.mean(tempoDiff)

    return torch.mean(batchDiff)

def spatioDiffFunc(X,Y):
    if len(X.shape) !=4:
        print('wrong!!!')
    if X.shape != Y.shape:
        print('wrong!!!')

    BS = X.shape[0]
    C = X.shape[1]

    batchDiff = torch.zeros((BS, 1))
    batchDiff = batchDiff.cuda(gpu_id)
    kernel = [[-0.125, -0.125, -0.125],
              [-0.125, 1, -0.125],
              [-0.125, -0.125, -0.125]]
    kernel = torch.FloatTensor(kernel)
    kernel = kernel.cuda(gpu_id)
    for i in range(BS):
        spatioDiff = torch.zeros((C, 1))
        spatioDiff = spatioDiff.cuda(gpu_id)
        for j in range(C):

            X_pad = torch.nn.functional.pad(X[i, j, :, :], (1, 1, 1, 1), "constant", 0)
            Y_pad = torch.nn.functional.pad(Y[i, j, :, :], (1, 1, 1, 1), "constant", 0)

            XX = torch.as_strided(X_pad, size=(512, 512, 3, 3), stride=(514, 1, 514, 1))
            YY = torch.as_strided(Y_pad, size=(512, 512, 3, 3), stride=(514, 1, 514, 1))
            dX_dbZ_pad = torch.tensordot(XX, kernel, [(2, 3), (0, 1)])
            dY_dbZ_pad = torch.tensordot(YY, kernel, [(2, 3), (0, 1)])

            spatioDiff[j] = criterion(dX_dbZ_pad, dY_dbZ_pad)

        batchDiff[i] = torch.mean(spatioDiff)
    return torch.mean(batchDiff)





class Conv_Block(torch.nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Conv_Block, self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.norm1=torch.nn.BatchNorm2d(output_channel)
        self.act1=torch.nn.LeakyReLU()
        self.conv2=torch.nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.norm2=torch.nn.BatchNorm2d(output_channel)
        self.act2=torch.nn.LeakyReLU()

    def forward(self,x):
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.norm2(x)
        res=self.act2(x)

        return res

class DownSample_Block(torch.nn.Module):
    def __init__(self):
        super(DownSample_Block, self).__init__()
        self.pooling1=torch.nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        res=self.pooling1(x)
        return res

class UpSample_Block(torch.nn.Module):
    def __init__(self,channnel):
        super(UpSample_Block, self).__init__()
        self.layer=torch.nn.Conv2d(in_channels=channnel,out_channels=channnel//2,kernel_size=1,stride=1,padding=0)
    def forward(self,x,ex):
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        res=self.layer(x)
        return torch.cat((res,ex),dim=1)

class MyUnet(torch.nn.Module):
    def __init__(self):
        super(MyUnet, self).__init__()
        self.Convlayer1=Conv_Block(20,32)#20->16 此处需要对应修改
        self.Down1=DownSample_Block()
        self.Convlayer2 = Conv_Block(32, 64)
        self.Down2 = DownSample_Block()
        self.Convlayer3 = Conv_Block(64, 128)
        self.Down3 = DownSample_Block()
        self.Convlayer4 = Conv_Block(128, 256)
        self.Down4 = DownSample_Block()
        self.Convlayer5 = Conv_Block(256, 512) #至此到了最底层
        self.Up1 = UpSample_Block(512)
        self.Convlayer6 = Conv_Block(512, 256)
        self.Up2 = UpSample_Block(256)
        self.Convlayer7 = Conv_Block(256, 128)
        self.Up3 = UpSample_Block(128)
        self.Convlayer8 = Conv_Block(128, 64)
        self.Up4 = UpSample_Block(64)
        self.Convlayer9 = Conv_Block(64, 32)
        self.FinalConv = torch.nn.Conv2d(32,20,3,1,1)

    def forward(self,x):
        C1=self.Convlayer1(x) #20->16 此处需要对应修改
        D1=self.Down1(C1)
        C2=self.Convlayer2(D1) #16->64
        D2=self.Down2(C2)
        C3=self.Convlayer3(D2) #64->128
        D3=self.Down3(C3)
        C4=self.Convlayer4(D3) #128->256
        D4=self.Down4(C4)
        C5=self.Convlayer5(D4) #256->512
        U1=self.Up1(C5,C4) #512->256->512
        C6=self.Convlayer6(U1) #512->256
        U2 = self.Up2(C6, C3)  # 256->128->256
        C7=self.Convlayer7(U2) #256->128
        U3 = self.Up3(C7, C2)  # 128->64->128
        C8=self.Convlayer8(U3) #128->64
        U4 = self.Up4(C8, C1)  # 64->32->64
        C9=self.Convlayer9(U4) #64->32
        Final=self.FinalConv(C9)
        res=torch.sigmoid(Final)

        return res

myUnet=MyUnet()
myUnet.cuda(gpu_id)
criterion = torch.nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM()




log_var_a = torch.zeros(1,).cuda(gpu_id)
log_var_b = torch.zeros(1,).cuda(gpu_id)
log_var_c = torch.zeros(1,).cuda(gpu_id)
log_var_a.requires_grad = True
log_var_b.requires_grad = True
log_var_c.requires_grad = True

# Initialized standard deviations (ground truth is 10 and 1):
std_1 = torch.exp(log_var_a)**0.5
std_2 = torch.exp(log_var_b)**0.5
std_3 = torch.exp(log_var_c)**0.5
params = ([p for p in myUnet.parameters()] + [log_var_a] + [log_var_b] + [log_var_c])




optimizer = optim.Adam(params, lr=0.00001)


if __name__ == '__main__':
    print('STRUENT')
    for epoch in range(10):
        myUnet.train()
        for i, data in enumerate(MyTrainLoader, 0):
            inputs, labels = data  # shape (BS,C,H,W)
            inputs = inputs.cuda(gpu_id)
            labels = labels.cuda(gpu_id)
            y_pred = myUnet(inputs)
            loss = torch.exp(-log_var_a) * (1 - ssim_loss(y_pred, labels)) + \
                   torch.exp(-log_var_b) * spatioDiffFunc(y_pred,labels) + \
                   torch.exp(-log_var_c) * tempodiffFunc(y_pred, labels) + \
                   log_var_a + log_var_b + log_var_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
