# -*- coding: utf-8 -*-
# 來源 : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 此範例是 CNN 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 這裡面是初始化等等要用在forward的各個layer層
        ############################################
        # # Kernel
        # # in (30*30*1)
        # self.conv1 = nn.Conv2d(1, 6, 3)   #(28*28*6) pool (14*14*6)
        # self.conv2 = nn.Conv2d(6, 16, 3)  #(12*12*6) pool (6*6*16)
        # # Affine operation
        # self.fc1 = nn.Linear(16*6*6, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        ############################################
        # # 官方教學設定
        # # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)    
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        ############################################

    def forward(self, x):  # 這個是一定要override的
        # 這裡面是寫 各個layer層 彼此是如何傳遞的
        # 這個架構運作方式
        # input(32*32) -> conv2d((32-5+1)*28*6) -> relu -> maxpool2d((28/2)*14*6) -> conv2d(10*10*16) -> relu -> maxpool2d(5*5*16)
        #   -> flatten -> linear -> relu -> linear -> relu -> linear
        #   -> MSELoss
        #   -> loss
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 這個也是 Max pooling  因為(2, 2)是正方形 所以可以簡化成一個數字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   
        # x = x.view(-1, self.num_flat_features(x))  # 不知道為什麼作者把下面一行 換成這個 # 意思應該是一樣的都是整形
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
       
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    # 初始化 NN
    net = Net()
    print("net construct :", net)

    ###################################################
    # net.parameters 會把全部可訓練的參數抓出來
    params = list(net.parameters())
    print(len(params))
    print("conv1's .weight :",params[0].size())
    for n in range(len(params)):
        print(params[n].size())

    ###################################################
    # <<丟入資料產生結果>>
    # 隨機產出一筆資料 丟進 NN 看看結果
    # 這個大小要符合架構的大小 
    # 好像可以接受一點誤差 ?? 作者的設定可以丟 (1,1,30,30)(正確) (1,1,32,32) 進去
    #                       官方的設定可以丟 (1,1,32,32)(正確) (1,1,34,34) 進去
    input = torch.randn(1,1,32,32)
    output = net(input)
    print('NN output 1 :', output)

    # <<歸零緩衝>>
    # 這個是backward之前一定要做的 要把"參數、grad緩衝區歸零"
    net.zero_grad()
    print('grad 1 :', net.conv1.bias.grad)  # None
    output.backward(torch.randn(1, 10))
    print('grad 2 :', net.conv1.bias.grad)
    # 注意上面只有backward(只有計算梯度) 沒有更新權重 所以print出來的東西會一樣
    output = net(input)
    print('NN output 2 :', output)

    ###################################################
    # 接下來是一系列的

    # <<示範怎麼計算 loss>>
    # 隨機產生兩筆資料 
    # 一筆當作丟進NN的輸入(input) 產出output
    # input直接沿用上面的
    output = net(input)
    # 一筆當作預期結果
    # 我們要使用 view() 來將預測的結果與標準答案 size 對齊 ?? 不懂
    target = torch.randn(10).view(1, -1)

    # 確認 output 的大小跟 target的大小一樣 (才能比較)
    print(output.size())  # torch.Size([1, 10])
    print(target.size())

    # 把 output 跟 target 丟入 loss計算公式計算結果
    # Loss function
    # EX:均方差(Mean square error, MSE)、平均絕對值誤差(Mean absolute error, MAE)、交叉熵(Cross-entropy) 
    criterion = nn.MSELoss()
    loss = criterion(output, target)  # output 會帶NN的東西進去 所以loss.backward() 才會更新grad 
    print("loss 1 :",loss)
    print("MSELoss :",loss.grad_fn)  # MSELoss
    print("Linear :",loss.grad_fn.next_functions[0][0])  # Linear
    print("ReLU :",loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # <<使用 loss 進行 Backprop>>
    # 通過計算預測結果與正解的距離（即 loss function），來反向地更新我們模型的權重
    # 一樣先歸零grad 
    net.zero_grad()  # 所以下面印出來才全部都是0
    print('grad Before :', net.conv1.bias.grad)
    # 再Backprop
    loss.backward()
    print('grad After :', net.conv1.bias.grad)  # 可以看到與 grad 2 計算出來的結果不一樣
    
    # 這裡也還沒更新權重 因此結果還是一樣的
    output = net(input)
    print('NN output 3 :', output)

    #<<更新權重>>
    ############################ 2 選 1 執行
    # 方法1 以下範例是實作 SGD
    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)
    ############################
    # 方法2 torch.optim 裡面有很多方法可以更新權重 EX:SGD, Nesterov-SGD, Adam, RMSProp
    import torch.optim as optim
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.step() # 執行這個才會更新參數
    ############################

    # 再丟入一樣的 input
    output = net(input)
    print('NN output 4 :', output)
    # 再次計算loss
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print("loss 2 :",loss) # 會發現loss變小了






