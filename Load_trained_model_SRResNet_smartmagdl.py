import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import matplotlib as mlp
import torch
from torch import nn
mlp.use('TkAgg')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class SRResNet(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )

        self.upconv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 1, kernel_size=9, stride=1, padding=4)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        res = self.residual_blocks(x)
        x = self.relu(self.upconv1(x))
        x = self.upconv2(x)
        return x

if __name__ == '__main__':
    data_set = 25
    path_label_test='/path_to_test_labels'
    path_data_test='/path_to_test_data'
    path_data_interpolation = '/path_to_test_interpolation'# optional
    all_simulations = []
    all_theory = []
    all_simulations_test = []
    all_theory_test = []
    all_interpolation = []
    train_pic_size = 300
    theory_pic_size = 300

    for j in range(1, 101):
        mat3 = scipy.io.loadmat(path_label_test  +'TheorB_'+ str(int(j)) + '.mat') #Btheor_tot_
        Btot_theory_test = mat3['B_ground_truth']
        all_theory_test.append(Btot_theory_test)
        mat4 = scipy.io.loadmat(path_data_interpolation +'SimA_'+ str(int(j)) + '.mat')

        Btot_sim_test = mat4['Bintr']
        all_simulations_test.append(Btot_sim_test)
        mat5 = scipy.io.loadmat(path_data_interpolation +'SimA_'+ str(int(j)) + '.mat')
        Btot_interp= mat5['Bintr']
        all_interpolation.append(Btot_interp)

    all_interpolation = 1e6*np.array(all_interpolation)
    Scaler = MinMaxScaler()
    x_test = np.array(all_simulations_test).reshape(100, 1,train_pic_size,train_pic_size)
    y_test = np.array(all_theory_test).reshape(-1, 1)
    X_interp = np.array(all_interpolation).reshape(100,train_pic_size,train_pic_size)
    x_test_check = Scaler.fit_transform(x_test.reshape(-1, 1))
    y_test_check = Scaler.fit_transform(y_test.reshape(-1, 1))
    x_test = x_test_check.reshape(100, 300, 300)
    y_test = y_test_check.reshape(100, 300, 300)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('/path_to_trained_model', map_location=torch.device('cpu'))
    generator = SRResNet().to(device)
    generator.load_state_dict(checkpoint)
    generator.eval()
    low_res_image = X_test.reshape(100,train_pic_size,train_pic_size) # for CNN
    low_res_image = torch.tensor(low_res_image, dtype=torch.float).unsqueeze(0).to(device)  # Shape: (1, 1, 33, 33)
    all_pred = []
    start_time = time.time()
    for k in range(0, 100):
        with torch.no_grad():
            high_res_image = generator(low_res_image[0, k].unsqueeze(0).unsqueeze(0))
            print("The run time of a singlw prediction is: " + str(time.time() - start_time))
            all_pred.append(high_res_image.squeeze(0).squeeze(0).cpu().numpy())
    all_pred = np.array(all_pred)
    x = np.linspace(0, 30, 300)
    y = np.linspace(0, 30, 300)
    X, Y = np.meshgrid(x, y)
    for run in range(0, 100):
        fig, ax = plt.subplots()
        plt.title('test case:' + str(run+1) + ' prediction')
        cs = ax.contourf(X, Y, Scaler.inverse_transform(all_pred[run]), levels=100, cmap='jet')  # high_res_image[0]
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
        fig, ax = plt.subplots()
        plt.title('test case:' + str(run+1) + ' simulation')
        cs = ax.contourf(X, Y,
                         Scaler.inverse_transform(x_test[run].reshape(-1, 1)).reshape(train_pic_size, train_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
        fig, ax = plt.subplots()
        plt.title('test case:' + str(run+1) + ' theory')
        cs = ax.contourf(X, Y,
                         Scaler.inverse_transform(y_test[run].reshape(-1, 1)).reshape(theory_pic_size, theory_pic_size),
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
        fig, ax = plt.subplots()
        plt.title('test case:' + str(run+1) + ' interpolation')
        cs = ax.contourf(X, Y,1e6*X_interp[run],
                         levels=100, cmap='jet')
        bar = plt.colorbar(cs)
        plt.xlabel('$X[m]$', size=17)
        plt.ylabel('$Y[m]$', size=17)
        bar.ax.set_ylabel('$B[\mu T]$', size=17)
    print('Done!')
