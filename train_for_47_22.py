import os

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.dataloader import myDataSet
from utils.utils import *

init_seed()

# hyperparameters
args = argparse()
args.batch_size = 32
args.epochs = 5000
args.learning_rate = 0.0003

# get the dataset
trainData = myDataSet()
valData = myDataSet(train=False, vaild=False, extra=True)
trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, drop_last=True)
valLoader = DataLoader(valData, batch_size=args.batch_size)

save_model_name = 'resnet18_attention_conv1d_'

if not os.path.exists('output/' + save_model_name[:-1]):
    os.mkdir('output/' + save_model_name[:-1])
    os.mkdir('output/' + save_model_name[:-1] + '/graph')
    os.mkdir('output/' + save_model_name[:-1] + '/graph/train')
    os.mkdir('output/' + save_model_name[:-1] + '/graph/valid')

output_dir = 'output/' + save_model_name[:-1] + '/'
# initialize the model
model = resnet18_with_attention_conv1d().to(args.device)
if os.path.exists(output_dir + save_model_name + 'best.pth'):
    load_weights(model, output_dir + save_model_name + 'best.pth')
model.cuda() if torch.cuda.is_available() else model.cpu()
# ----------------------------------------------------------------


# initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
criterion = myLoss(alpha=0.5).to(args.device)
# -----------------------------------------------------------------


# initialize variables
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
lr = []
valid_best = 50
# ---------------------------------------------------------------------


# train

try:
    print('start train')
    for epoch in range(args.epochs):

        # ====================train==============================
        print('\n----------------train-best:' + str(valid_best) + '-------------------')
        model.train()
        train_epoch_loss = []
        for idx, (x1, x2, y) in enumerate(trainLoader):

            x1, x2, y = Variable(x1).cuda(), Variable(
                x2).cuda(), Variable(y).cuda()
            yHat = model((x1, x2))
            loss = criterion(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % 500 == 0:
                print("epoch={}/{},batch={}/{} of train, loss={:.2f}".format(
                    epoch + 1, args.epochs, idx + 1, len(trainLoader), loss.item()))
        lr.append(scheduler.get_last_lr())
        if epoch % 10 == 0:
            scheduler.step()
        print("epoch={}/{} of train, average loss={:.2f}".format(
            epoch + 1, args.epochs, np.average(train_epoch_loss)))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        print('-------------------valid-----------------')
        model.eval()
        valid_epoch_loss = []
        for idx, (x1, x2, y) in enumerate(valLoader):
            x1[:, 2, :, :] = x1[:, 2, :, :]
            x2[:, 2] = x2[:, 2]
            x1, x2, y = Variable(x1).cuda(), Variable(
                x2).cuda(), Variable(y).cuda()
            yHat = model((x1, x2))
            loss = criterion(yHat, y)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            if idx % 500 == 0:
                print("epoch={}/{},batch={}/{} of valid, loss={:.2f}".format(
                    epoch + 1, args.epochs, idx + 1, len(valLoader), loss.item()))
        print("epoch={}/{} of valid, average loss={:.2f}".format(
            epoch + 1, args.epochs, np.average(valid_epoch_loss)))
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        # save the optimal weights
        if np.average(valid_epoch_loss) < valid_best:
            valid_best = np.average(valid_epoch_loss)
            torch.save(model.state_dict(), output_dir +
                       save_model_name + 'best.pth')

        if epoch + 1 in [150, 200, 300, 500, 1000, 1500, 2000, 2500, 3000]:
            torch.save(model.state_dict(), output_dir +
                       save_model_name + str(epoch + 1) + '.pth')

    # plot the loss during training
    print('finish')
    print('saving the last model')
    torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
    draw_loss(train_epochs_loss, valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
    draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')


except KeyboardInterrupt:
    print('stop')
    print('saving the last model')
    torch.save(model.state_dict(), output_dir + save_model_name + 'last.pth')
    draw_loss(train_epochs_loss, valid_epochs_loss, fig_name=output_dir + 'graph/train/loss.png')
    draw_lr(lr, fig_name=output_dir + 'graph/train/learning_rate.png')
