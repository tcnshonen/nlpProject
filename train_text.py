import os
import torch
import torch.nn.functional as F
import argparse
import torch.optim as optim
from time import perf_counter
from torch.utils.data import DataLoader, random_split

from utils.constants import *
from utils.dataset import MixDataset, my_collate
from networks.models import TextOffsetModel, Autoencoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--save_frequency', default=10, type=int)
    args = parser.parse_args()


    print('Loading Autoencoder')
    autoencoder = Autoencoder().to(device)
    #autoencoder.load_state_dict(torch.load('weights/model1.pth'))
    autoencoder.eval()

    print('Creating Model')
    text_model = TextOffsetModel().to(device)
    optimizer = optim.Adam(text_model.parameters(), lr=args.lr)

    print('Loading Data')
    dataset = MixDataset('../mario')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=my_collate)


    for epoch in range(1, args.epochs+1):
        length = len(train_loader)
        t0 = perf_counter()
        text_model.train()
        for i, data in enumerate(train_loader):
            if data is None:
                continue
            img1, _, img2, _, sent, target = data
            if len(img1) == 1:
                continue

            with torch.no_grad():
                embedding1 = autoencoder.flatten_forward(img1)
                embedding2 = autoencoder.flatten_forward(img2)

            optimizer.zero_grad()
            pred = text_model(sent, embedding1, embedding2)
            loss = F.binary_cross_entropy(pred, target)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f} - {}s'.format(
                epoch, i + 1, length, 100. * (i + 1) / length, loss.item(),
                int(perf_counter() - t0)), end='\r')
            torch.cuda.empty_cache()


        print()
        length = len(test_loader)
        t0 = perf_counter()
        total_loss, acc = 0., 0.
        count1, count2 = 0, 0
        text_model.eval()
        for i, data in enumerate(test_loader):
            if data is None:
                continue
            img1, _, img2, _, sent, target = data
            count1 += 1
            count2 += len(img1)

            with torch.no_grad():
                embedding1 = autoencoder.flatten_forward(img1)
                embedding2 = autoencoder.flatten_forward(img2)

            pred = text_model(sent, embedding1, embedding2)
            total_loss += F.binary_cross_entropy(pred, target).item()
            _, pred_idx = torch.max(pred, 1)
            _, target_idx = torch.max(target, 1)
            acc += sum([i == j for i, j in zip(pred_idx, target_idx)]).item()


            print('Train Epoch: {} [{}/{} ({:.1f}%)] - {}s'.format(
                epoch, i + 1, length, 100. * (i + 1) / length,
                int(perf_counter() - t0)))
            torch.cuda.empty_cache()

        print()
        print('Loss: {:.4f}\tAcc: {:.2f}%'.format(total_loss / count1, acc / count2))
        print()

        if epoch % args.save_frequency == 0:
            torch.save(text_model.state_dict(), 'weights/text_offset_model{}.pth'.format(epoch))


    # print('Creating Model')
    # mix_model = MixModel().to(device)
    # optimizer = optim.Adam(mix_model.parameters(), lr=args.lr)
    #
    # print('Loading Data')
    # dataset = MixDataset('../mario')
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                           shuffle=True, collate_fn=my_collate)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
    #                          shuffle=False, collate_fn=my_collate)
    #
    #
    # for epoch in range(1, args.epochs+1):
    #     length = len(train_loader)
    #     t0 = perf_counter()
    #     mix_model.train()
    #     for i, data in enumerate(train_loader):
    #         if i > 3: break
    #         if data is None:
    #             continue
    #         img1, ram1, img2, ram2, sent, target = data
    #
    #         optimizer.zero_grad()
    #         pred_ram1, pred_ram2, pred_cls = mix_model(img1, img2, sent)
    #         loss1 = F.cross_entropy(pred_ram1, ram1)
    #         loss2 = F.cross_entropy(pred_ram2, ram2)
    #         loss3 = F.binary_cross_entropy(pred_cls, target)
    #         loss = loss1 + loss2 + loss3
    #         loss.backward()
    #         optimizer.step()
    #
    #         print('Train Epoch: {} [{}/{} ({:.1f}%)]\tRAM1 Loss: {:.4f}\tRAM2 Loss: {:.4f}\tCLS Loss: {:.4f} - {}s'.format(
    #             epoch, i + 1, length, 100. * (i + 1) / length, loss1.item(),
    #             loss2.item(), loss3.item(), int(perf_counter() - t0)), end='\r')
    #         torch.cuda.empty_cache()
    #
    #     length = len(test_loader)
    #     t0 = perf_counter()
    #     mix_model.eval()
    #     total_loss1, total_loss2, total_loss3 = 0., 0., 0.
    #     count = 0
    #     for i, data in enumerate(test_loader):
    #         if data is None:
    #             continue
    #         count += 1
    #         img1, ram1, img2, ram2, sent, target = data
    #         pred_ram1, pred_ram2, pred_cls = mix_model(img1, img2, sent)
    #         total_loss1 += F.cross_entropy(pred_ram1, ram1).item()
    #         total_loss2 += F.cross_entropy(pred_ram2, ram2).item()
    #         total_loss3 += F.binary_cross_entropy(pred_cls, target).item()
    #
    #         print('Test Epoch: {} [{}/{} ({:.1f}%)] - {}s'.format(
    #             epoch, i + 1, length, 100. * (i + 1) / length,
    #             int(perf_counter() - t0)), end='\r')
    #
    #     print()
    #     print('Test RAM1 Loss: {}\t Test CLS Loss: {}'.format(
    #         (total_loss1 + total_loss2) / (count * 2), total_loss3 / count))
    #     print()
    #
    #
    #     if epoch % args.save_frequency == 0:
    #         torch.save(mix_model.state_dict(), 'weights/mix_model{}.pth'.format(epoch))


    # print('Loading Autoencoder')
    # autoencoder = Autoencoder().to(device)
    # autoencoder.load_state_dict(torch.load('weights/model1.pth'))
    # autoencoder.eval()

    # print('Creating Model')
    # text_model = TextModel().to(device)
    # optimizer = optim.Adam(text_model.parameters(), lr=args.lr)

    # print('Loading Data')
    # dataset = TextTrainingDataset('../mario')
    # loader = DataLoader(dataset, batch_size=args.batch_size,
    #                     shuffle=True, collate_fn=my_collate)



    # for epoch in range(1, args.epochs+1):
    #     length = len(loader)
    #     t0 = perf_counter()
    #     for i, (img1, ram1, img2, ram2, sent, target) in enumerate(loader):
    #         with torch.no_grad():
    #             embedding1 = autoencoder.flatten_forward(img1)
    #             embedding2 = autoencoder.flatten_forward(img2)
    #
    #         optimizer.zero_grad()
    #         pred = text_model(sent, embedding1, embedding2)
    #         loss = F.binary_cross_entropy(F.sigmoid(pred), target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f} - {}s'.format(
    #             epoch, i + 1, length, 100. * (i + 1) / length, loss.item(),
    #             int(perf_counter() - t0)), end='\r')
    #
    #         torch.cuda.empty_cache()
    #
    #     if epoch % args.save_frequency == 0:
    #         torch.save(text_model.state_dict(), 'weights/text_model{}.pth'.format(epoch))
