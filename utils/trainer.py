import torch
import torch.nn.functional as F
from time import perf_counter

from .constants import device


def train(epoch, net, opt, loader):
    net.train()
    length = len(loader)
    t0 = perf_counter()
    for i, (img1_target, ram1_target, img2_target, ram2_target, sent) in enumerate(loader):
        #img1_target = img1_target.to(device)
        #ram1_target = ram1_target.to(device, dtype=torch.long)
        #img2_target = img2_target.to(device)
        #ram2_target = ram2_target.to(device, dtype=torch.long)

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

        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tMemory Loss: {:.4f}\tImage Loss: {:.4f}\tSentence Loss: {:.4f} - {}s'.format(
            epoch, i + 1, length, 100. * (i + 1) / length,
            ram1_loss.item() + ram2_loss.item(), img1_loss.item() + img2_loss.item(),
            sen_loss.item(), int(perf_counter() - t0)), end='')

        del img1, img1_target, ram1_target, img2, img2_target, ram2_target
        torch.cuda.empty_cache()


def test():
    return
