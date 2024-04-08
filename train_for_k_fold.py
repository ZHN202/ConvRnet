import argparse
import os
import time

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import myDataSet_k_fold
from utils.utils import draw_loss, draw_lr, myLoss, get_model_name, get_model
from utils.utils import init_seed

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--start_fold", type=int, default=0)
parser.add_argument("--stop_fold", type=int, default=19)
parser.add_argument("--epochs", type=int, default=70)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)  ##0.0001
parser.add_argument("--output_dir_name", type=str, default='20-4-Fold-Dataset-1')
parser.add_argument("--ChooseModel", type=int, default=3)
args = parser.parse_args()

# initialize a random seed
init_seed(seed=2602)

# Hyperparameters


'''
Models to choose from:
1--otherModel.py   --->   Linear  MLP
2--otherModel.py   --->   Conv1d  
3--conv_1.py       --->   ConvRnet
4--conv_1.py       --->   ConvRnet_linear
5--conv_1.py       --->   ConvRnet_without_CBAM
6--conv_1.py       --->   ConvRnet_without_DM
7--conv_1.py       --->   ConvMLP
8--rbf.py          --->   RBF
9--rbfmlp.py       --->   RBF_MLP
'''


def train(fold):
    ChooseModel = args.ChooseModel
    save_model_name = get_model_name(ChooseModel)

    # fold = args.fold
    output_dir_name = args.output_dir_name + '/Fold-' + str(fold)
    writer = SummaryWriter('logs/' + save_model_name + '/Fold-' + str(fold))
    # get the dataset
    trainData = myDataSet_k_fold(dataProcess='MS', path=output_dir_name + '/train.txt')
    valData = myDataSet_k_fold(dataProcess='MS', path=output_dir_name + '/test.txt')

    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if not os.path.exists(output_dir_name + '/' + save_model_name[:-1]):
        os.mkdir(output_dir_name + '/' + save_model_name[:-1])
        os.mkdir(output_dir_name + '/' + save_model_name[:-1] + '/graph')
        os.mkdir(output_dir_name + '/' + save_model_name[:-1] + '/graph/train')
        os.mkdir(output_dir_name + '/' + save_model_name[:-1] + '/graph/test')
        os.mkdir(output_dir_name + '/' + save_model_name[:-1] + '/log')

    output_dir = output_dir_name + '/' + save_model_name[:-1] + '/'

    # initialize the model
    model = get_model(ChooseModel, args.device)

    model.cuda()
    # ----------------------------------------------------------------

    # initialize the optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = myLoss(alpha=1).to(args.device)

    # -----------------------------------------------------------------

    f = open(output_dir_name + '/' + save_model_name[:-1] + '/log/' + save_model_name + 'log.txt', 'w+')

    # training

    train_epochs_loss = []
    valid_epochs_loss = []
    lr = []
    prevalid = 50
    valid_best = 50
    cnt = 0
    # ---------------------------------------------------------------------
    try:
        print('start train')
        for epoch in range(args.epochs):
            strat = time.time()

            print('----------------train-------------------' + save_model_name)
            model.train()
            train_batch_loss = []
            for idx, (x1, x2, y) in enumerate(trainLoader):
                x1, x2, y = Variable(x1).cuda(), Variable(
                    x2).cuda(), Variable(y).cuda()
                yHat = model((x1, x2))

                loss = criterion(yHat * 50, y * 50)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())

            writer.add_scalar('Train Loss', np.average(train_batch_loss), epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            train_epoch_loss = np.average(train_batch_loss)
            print("Epoch={}/{},   Train loss={:.4f}\n".format(
                epoch + 1, args.epochs, train_epoch_loss))
            f.write("Epoch={}/{},   Train loss={:.4f}\n".format(
                epoch + 1, args.epochs, train_epoch_loss))
            print(str(time.time() - strat))
            # test
            print('-------------------valid-----------------' + str(valid_best))
            model.eval()
            valid_batch_loss = []
            for idx, (x1, x2, y) in enumerate(valLoader):
                x1[:, 2, :, :] = x1[:, 2, :, :]
                x2[:, 2] = x2[:, 2]
                x1, x2, y = Variable(x1).cuda(), Variable(
                    x2).cuda(), Variable(y).cuda()
                yHat = model((x1, x2))

                loss = criterion(yHat * 50, y * 50)

                valid_batch_loss.append(loss.item())
            writer.add_scalar('Test Loss', np.average(valid_batch_loss), epoch)
            valid_epoch_loss = np.average(valid_batch_loss)
            print("Epoch={}/{},   Val loss={:.4f}\n".format(
                epoch + 1, args.epochs, valid_epoch_loss))
            f.write("Epoch={}/{},   Val loss={:.4f}\n".format(
                epoch + 1, args.epochs, valid_epoch_loss))

            # adjust the learning rate

            scheduler.step(np.average(valid_batch_loss))

            train_epochs_loss.append(train_epoch_loss)
            valid_epochs_loss.append(valid_epoch_loss)

            # save the optimal weights
            if valid_epoch_loss < valid_best:
                valid_best = valid_epoch_loss
                torch.save(model.state_dict(), output_dir +
                           save_model_name + 'best.pth')

            if epoch + 1 in [100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000]:
                torch.save(model.state_dict(), output_dir +
                           save_model_name + str(epoch + 1) + '.pth')
            if optimizer.param_groups[0]['lr'] == 1e-8:
                if prevalid == valid_best:
                    cnt += 1
                else:
                    prevalid = valid_best
                    cnt = 0
            if cnt > 9:
                break

        # plot the loss during training
        print('finish')
        print('saving the last model')
        torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
        draw_loss(train_epochs_loss,
                  valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
        draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')
        plt.close()
        f.close()


    except KeyboardInterrupt:
        print('stop')
        print('saving the last model')
        f.close()
        torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
        draw_loss(train_epochs_loss,
                  valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
        draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')
        plt.close()


if __name__ == '__main__':
    for i in range(args.start_fold, args.stop_fold + 1):
        print('Start Training FOLD  ' + str(i))
        train(i)
