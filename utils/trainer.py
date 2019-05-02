import torch
import torch.nn.functional as F
from time import perf_counter

from .constants import device


class Trainer(object):
    def __init__(self, net, opt, loader):
        self.net = net
        self.opt = opt
        self.loader = loader
        self.length = len(self.loader)

    def forward(self, data):
        raise NotImplemented

    def log(self, epoch):
        raise NotImplemented

    def train(self, epoch):
        self.net.train()
        length = len(self.loader)
        t0 = perf_counter()

        for i, data in enumerate(self.loader):
            self.opt.zero_grad()
            loss = self.forward(data)
            loss.backward()
            self.opt.step()

            log(epoch, i, t0, loss.item())

            torch.cuda.empty_cache()


class AE_Trainer(Trainer):
    def __init__(self):
        super(AE_Trainer, self).__init__()

    def forward(self, data):
        img, ram = data
        ram_output = self.net(ram)
        loss = F.cross_entropy(ram_output, ram)
        return loss

    def log(self, epoch, i, t0, loss):
        print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f} - {}s'.format(
            epoch, i + 1, self.length, 100. * (i + 1) / self.length,
            loss, int(perf_counter() - t0)), end='\r')



def train(epoch, net, opt, loader):
    net.train()
    length = len(loader)
    t0 = perf_counter()

    for i, (img1_target, ram1_target, img2_target, ram2_target, sent) in enumerate(loader):

        indices = [i for i, x in enumerate(sent) if not all(x == torch.zeros(8, dtype=torch.long, device=device))]
        img1_target, ram1_target = img1_target[indices], ram1_target[indices]
        img2_target, ram2_target = img2_target[indices], ram2_target[indices]
        sent = sent[indices]

        if len(indices) == 1:
            continue

        img1, img2 = img1_target.clone(), img2_target.clone()

        opt.zero_grad()
        img1_output, ram1_output, img2_output, ram2_output, diff, embedding = net(img1, img2, sent)
        img1_loss = F.mse_loss(img1_output, img1_target)
        ram1_loss = F.cross_entropy(ram1_output, ram1_target)
        img2_loss = F.mse_loss(img2_output, img2_target)
        ram2_loss = F.cross_entropy(ram2_output, ram2_target)
        sen_loss = F.mse_loss(embedding, diff)
        loss = img1_loss + ram1_loss + img2_loss + ram2_loss + sen_loss
        loss.backward()
        opt.step()

        print('\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tMemory Loss: {:.4f}\tImage Loss: {:.4f}\tSentence Loss: {:.4f} - {}s'.format(
            epoch, i + 1, length, 100. * (i + 1) / length,
            ram1_loss.item() + ram2_loss.item(), img1_loss.item() + img2_loss.item(),
            sen_loss.item(), int(perf_counter() - t0)), end='')

        del img1, img1_target, ram1_target, img2, img2_target, ram2_target
        torch.cuda.empty_cache()


def get_embeddings(net, loader, log=True):
    net.eval()
    length = len(loader)
    t0 = perf_counter()

    embeddings = []
    embedding_nums = []
    embedding_actions = []

    for i, (img, _, (num, action)) in enumerate(loader):

        embedding = net.flatten_forward(img)
        embeddings.extend(embedding.cpu().detach().numpy())
        embedding_nums.extend(num)
        embedding_actions.extend(action)

        print('{}/{} ({:.1f}%) - {}s'.format(
            i + 1, length, 100. * (i + 1) / length,
            int(perf_counter() - t0)), end='\r')

        del img, num, action, embedding
        torch.cuda.empty_cache()

    return embeddings, embedding_nums, embedding_actions
