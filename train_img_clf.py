import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from models import LeNet

def mnist_loader(num_classes, batch_size=64):
    tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=tforms)
    idx = train_set.targets < num_classes
    train_set.data = train_set.data[idx]
    train_set.targets = train_set.targets[idx]

    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=tforms)
    idx = test_set.targets < num_classes
    test_set.data = test_set.data[idx]
    test_set.targets = test_set.targets[idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader

class Trainer():
    def __init__(self, model, train_loader, test_loader, num_epochs=20, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
        self.loss_function = nn.CrossEntropyLoss()

    def train(self):
        max_acc = 0
        for epoch in range(self.num_epochs):
            print('\nEpoch {}'.format(epoch))
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test()

            print('Training set: Average loss: {:.6f}'.format(train_loss))
            print('Training set: Accuracy: {:.2f}'.format(train_acc))
            print('Test set: Average loss: {:.6f}'.format(test_loss))
            print('Test set: Accuracy: {:.2f}'.format(test_acc))

            if test_acc > max_acc:
                max_acc = test_acc
                print('Saving model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, './saved_models/model.pt')

    def train_epoch(self):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for (data, target) in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            train_loss.update(loss.item())
            acc = compute_accuracy(output, target)
            train_acc.update(acc)
            loss.backward()
            self.optimizer.step()

        return train_loss.get_avg(), train_acc.get_avg()

    def test(self):
        self.model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            for data, target in self.test_loader:            
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)         
                loss += self.loss_function(output, target).item()

                cur_acc = compute_accuracy(output, target)
                acc += cur_acc
        
            loss, acc = loss / len(self.test_loader), acc / len(self.test_loader)
    
        return loss, acc

def compute_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.shape[0]

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top1_acc = correct[:1].view(-1).float().sum(0, keepdim=True) * 100.0 / batch_size

    return top1_acc.item()

class AverageMeter():
    """Computes and stores the average and current value
    
    Taken from the Torch examples repository:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

if __name__ == '__main__':
    mnist_train, mnist_test = mnist_loader(num_classes=2)
    model = LeNet(in_channels=1, num_classes=2)
    trainer = Trainer(model, mnist_train, mnist_test)
    trainer.train()