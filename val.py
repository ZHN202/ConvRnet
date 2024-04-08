import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.dataloader import myDataSet_k_fold
from utils.utils import *

init_seed()

# Hyperparameters
args = argparse()
args.batch_size = 1
args.epochs = 1

'''
models to choose from
1--otherModel.py   --->   Linear  MLP
2--otherModel.py   --->   Conv1d  
3--conv_1.py       --->   ConvRnet
4--conv_1.py       --->   ConvRnet_linear
5--conv_1.py       --->   ConvRnet_without_CBAM
6--conv_1.py       --->   ConvRnet_without_DM
'''
fold = 1
ChooseModel = 7


if ChooseModel == 1:
    save_model_name = 'Linear_MS_'
elif ChooseModel == 2:
    save_model_name = 'Conv1d_MS_'
elif ChooseModel == 3:
    save_model_name = 'ConvRnet_MS_'
elif ChooseModel == 4:
    save_model_name = 'ConvRnet_Linear_MS_'
elif ChooseModel == 5:
    save_model_name = 'ConvRnet_CBAM_MS_'
elif ChooseModel == 6:
    save_model_name = 'ConvRnet_DM_MS_'
elif ChooseModel == 7:
    save_model_name = 'ConvMLP_MS_'
else:
    print('please choose the right model')

output_dir_name = '20-4-Fold-Dataset-1/Fold-' + str(fold)

# get the dataset
# valData = myDataSet(dataProcess='MS',train=False, vaild=False, extra=True)
valData = myDataSet_k_fold(dataProcess='MS', path=output_dir_name + '/test.txt')  #
valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=False)

output_dir = output_dir_name + '/' + save_model_name[:-1] + '/'
model_name = output_dir + save_model_name + 'best.pth'
output_dir = output_dir + 'graph/test/'

# initialize the model
if ChooseModel == 1:
    model = Linear().to(args.device)
elif ChooseModel == 2:
    model = Conv1d().to(args.device)
elif ChooseModel == 3:
    model = ConvRnet().to(args.device)
elif ChooseModel == 4:
    model = ConvRnet_linear().to(args.device)
elif ChooseModel == 5:
    model = ConvRnet_without_CBAM().to(args.device)
elif ChooseModel == 6:
    model = ConvRnet_without_CBAM().to(args.device)
elif ChooseModel == 7:
    model = ConvMLP().to(args.device)
else:
    print('please choose the right model')

load_weights(model, model_name)
model.cuda()
# ----------------------------------------------------------------


# initialize variables
RMSE_val = []
MSE_val = []
MAE_val = []
gt = []
pred = []
errors = []
# ---------------------------------------------------------------------


# 

try:
    model.eval()
    with torch.no_grad():
        print('start valid')

        for epoch in range(args.epochs):
            # =====================valid============================
            print('-------------------valid-----------------')

            valid_epoch_loss = []
            for idx, (x1, x2, y) in enumerate(valLoader):
                if (idx) < 2700:
                    x1, x2 = Variable(x1).cuda(), Variable(
                        x2).cuda()
                    print(x1.shape, x2)
                    yHat = model((x1, x2)).cpu()
                    yHat = torch.clamp(yHat, min=0) * 50
                    y = y * 50
                    gt.append(y[0].numpy())
                    pred.append(yHat[0].numpy())
                    inp = x2[0].cpu().numpy()

                    mse, rmse, mae = calculate_evaluation_metrics(y[0].numpy(), yHat[0].numpy())

                    RMSE_val.append(rmse)
                    MSE_val.append(mse)
                    MAE_val.append(mae)
                    error = yHat[0] - y[0]
                    errors.append(error)
                    print('------------------batch:{}---------------'.format(idx + 1))
                    print(' | real | pred | error | mse | rmse | mae |')
                    print("x|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                        y[0][0], yHat[0][0], error[0], mse[0], rmse[0], mae[0]))
                    print("y|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|\n".format(
                        y[0][1], yHat[0][1], error[1], mse[1], rmse[1], mae[1]))

    # plot the loss during training
    print(' | mse | rmse | mae |')
    print("|{:.2f}|{:.2f}|{:.2f}|".format(
        np.average(MSE_val), np.average(RMSE_val), np.average(MAE_val)))

    draw_valid(MSE_val, RMSE_val, MAE_val, fig_name=output_dir + 'valid.png')
    draw_error(gt, pred, fig_name=output_dir + 'error.png')


except KeyboardInterrupt:
    draw_valid(MSE_val, RMSE_val, MAE_val, fig_name=output_dir + 'valid.png')
    draw_error(gt, pred, fig_name=output_dir + 'error.png')
    print('stop')
