import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import scipy.io
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError

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
        x = x + res
        x = self.relu(self.upconv1(x))
        x = self.upconv2(x)
        return x

def content_loss(output, target):
    return F.mse_loss(output, target)

def train():
    path_labels = '/path_to_labels'
    path_data = '/path_to_data'

    all_simulations = []
    all_theory = []
    train_pic_size = 300
    theory_pic_size = 300

    for i in range(1, 1001):
        mat1 = scipy.io.loadmat(path_labels +'TheorB_' + str(int(i)) + '.mat')
        Btot_theory_i =mat1['B_ground_truth']
        all_theory.append(Btot_theory_i)
        mat2 = scipy.io.loadmat(path_data +'SimA_'+ str(int(i)) + '.mat')
        Btot_sim_i =mat2['Bintr']
        all_simulations.append(Btot_sim_i)

    X_train = np.array(all_simulations)
    y_train = np.array(all_theory)
    X_train = X_train.reshape(1000, 1 ,train_pic_size,train_pic_size)
    Scaler = MinMaxScaler()
    y_train = y_train.reshape(1000, 1 , theory_pic_size,theory_pic_size)
    X_train_check = Scaler.fit_transform(X_train.reshape(-1, 1))
    y_train_check = Scaler.fit_transform(y_train.reshape(-1, 1))
    X_train = X_train_check.reshape(1000, 1, 300, 300)
    y_train = y_train_check.reshape(1000, 1, 300, 300)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_ratio = 0.8
    train_size = int(train_ratio * X_train.shape[0])
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    validation_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SRResNet().to(device)
    train_losses = []
    validation_losses = []
    optimizer = optim.Adam(generator.parameters(), lr=1e-4)


    def train_step(engine, batch):
        low_res_images, high_res_images = batch
        low_res_images = low_res_images.to(device)
        high_res_images = high_res_images.to(device)
        optimizer.zero_grad()
        generated_images = generator(low_res_images)
        loss = content_loss(generated_images, high_res_images)
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(engine, batch):
        low_res_images, high_res_images = batch
        low_res_images = low_res_images.to(device)
        high_res_images = high_res_images.to(device)
        with torch.no_grad():
            generated_images = generator(low_res_images)
        return generated_images, high_res_images

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    mse_metric = MeanSquaredError()
    mse_metric.attach(evaluator, 'mse')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        train_loss = metrics['mse']
        train_losses.append(train_loss)
        print(f"Epoch {engine.state.epoch} - Avg loss: {engine.state.output:.4f}, train mse: {train_loss:.4f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        validation_loss = metrics['mse']
        validation_losses.append(validation_loss)
        print(f"validation mse: {validation_loss:.4f}")

    num_of_epochs=200
    trainer.run(train_loader, max_epochs=num_of_epochs)
    torch.save(generator.state_dict(), os.path.join(
        os.getcwd() + '/' + f'{time.time():.3}' + '_SRResNet_16_Blocks_'+str(num_of_epochs)+'_epochs_simA_theoryB' + '.pth'))

if __name__=='__main__':
    train()
