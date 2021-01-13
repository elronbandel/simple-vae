"""Training procedure for VAE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model
import matplotlib.pyplot as plt


def train(vae, trainloader, optimizer, epoch):
    vae.train()  # set to training mode
    total_loss = 0.0
    batches = 0
    for i, (inputs, _) in enumerate(trainloader):
        # inputs shape: BxCxHxW
        loss = vae(inputs)
        loss.backward()
        optimizer.step()
        vae.zero_grad()
        batches += 1
        total_loss += loss.item()
    epoch_mean_loss = total_loss / batches
    print(f'    Train ELBO:{epoch_mean_loss}')
    return epoch_mean_loss


def test(vae, testloader, filename, epoch):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        total_loss = 0.0
        batches = 0
        for inputs, _ in testloader:
            loss = vae(inputs)
            total_loss += loss.item()
            batches += 1
        test_mean_loss = total_loss / batches
        print(f'    Test ELBO:{test_mean_loss}')
        return test_mean_loss


def sample(vae, sample_size, filename, epoch):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        samples = vae.sample(sample_size).cpu()
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.)),  # dequantization
        transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
               + 'batch%d_' % args.batch_size \
               + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim, device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    train_loss, test_loss = [], []
    for epoch in range(args.epochs):
        print(f'--Epoch {epoch}--')
        train_loss.append(train(vae, trainloader, optimizer, epoch))
        test_loss.append(test(vae, testloader, filename, epoch))
        sample(vae, args.sample_size, filename, epoch)
    print(train_loss)
    plt.plot(train_loss, label='Train Elbo (Sum)')
    plt.legend()
    plt.plot(test_loss, label='Test Elbo (Sum)')
    plt.legend()
    plt.savefig(f'plots/elbo_plot_{args.dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=30)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
