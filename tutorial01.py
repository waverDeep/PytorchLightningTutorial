import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning import Trainer


# lightning module은 model을 정의하는 것이 아닌 System을 정의하는 것이라고 한다.


# system 정의하기
class LiteMNIST(pl.LightningModule):
    def __init__(self):
        super(LiteMNIST, self).__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    # it is inference logic
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    # it is training loop logic
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    # optimizer logic
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

    # data
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_train = DataLoader(mnist_train, batch_size=64)

    model = LiteMNIST()
    # trainer = Trainer(gpus=1)
    trainer = Trainer(gpus=3, num_nodes=1, strategy="ddp")
    trainer.fit(model, mnist_train)






# the trainer automates:
# - epoch and batch iteration
# - calling of optimzer.step(), backward, zero_grad()
# - calling of .eval(), enabling/disabling grads
# - checkpoint saving and loading
# - tensorboard(see loggers options)
# - Multi-gpu support ***
# - TPU
# - 16-bit precision amp support ***





