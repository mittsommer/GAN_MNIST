import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--epoch", default=30)
parser.add_argument("--device", default='cpu')
CFG = parser.parse_args()

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device(CFG.device)

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = MNIST('./data', train=True, download=True, transform=transformer)
test_data = MNIST('./data', train=False, download=True, transform=transformer)
train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=CFG.batch_size, shuffle=True)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(100, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear5 = nn.Linear(1024, 28 * 28)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = torch.tanh(self.linear5(x))
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


G = generator().to(device)
D = discriminator().to(device)
loss_function = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.001, betas=(.5, .999))
d_optimizer = optim.Adam(D.parameters(), lr=0.001, betas=(.5, .999))


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        '''
        Training step:
        1. feed real_img into discriminator
        2. calculate the loss of the real_img
        3. generate a fake_img
        4. feed fake_img into discriminator
        5. calculate the loss of the fake_img
        6. add real_loss and fake_loss together = loss of discriminator
        7. back propagation discriminator
        8. generate a new fake_img
        9. back propagation generator
        '''
        img_num = data.size()[0].to(device)
        # ----------train discriminator----------
        real_img = data.view(img_num, -1)
        real_label = torch.ones(img_num, 1)
        real_prd = D(real_img)
        d_loss_real = loss_function(real_prd, real_label)

        latent = torch.randn(img_num, 100)
        fake_img = G(latent)  # generate fake img
        fake_prd = D(fake_img)  # predict fake img with discriminator
        fake_label = torch.zeros(img_num, 1)  # fake img label [0,0,...0]
        d_loss_fake = loss_function(fake_prd, fake_label)

        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # ----------train generator--------------
        latent = torch.randn(img_num, 100)
        fake_img = G(latent)
        fake_prd = D(fake_img)
        g_loss = loss_function(fake_prd, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] g_loss: {:.6f} d_loss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                g_loss.item(), d_loss.item()))
    save_image(fake_img.view(-1, 1, 28, 28), fp='./epoch{}.jpg'.format(epoch))


if __name__ == "__main__":
    for e in range(CFG.epoch):
        train(epoch=e)
