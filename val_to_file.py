import os

import numpy as np
import pandas as pd
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import myDataSet_k_fold
from utils.utils import *

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--start_fold", type=int, default=0)
parser.add_argument("--stop_fold", type=int, default=19)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dir_path", type=str, default=r'D:\。\Underground\sci\20-4-Fold-Dataset-1')
parser.add_argument("--output_path", type=str, default='output_new')
args = parser.parse_args()


#
def getData(ChooseModel, model_path, dataset_path, batch_size):
    # 获取数据集
    valData = myDataSet_k_fold(dataProcess='MS', path=dataset_path)
    valLoader = DataLoader(valData, batch_size=batch_size, shuffle=False)
    # 初始化模型
    model = get_model(ChooseModel, args.device)
    load_weights(model, model_path)
    model.cuda()
    model.eval()
    with torch.no_grad():
        print('start valid')

        MSE_V = []
        MSE_H = []
        MAE_V = []
        MAE_H = []
        gt = []
        for idx, (x1, x2, y) in enumerate(valLoader):

            x1, x2 = Variable(x1).cuda(), Variable(
                x2).cuda()
            yHat = model((x1, x2)).cpu()
            yHat = torch.clamp(yHat, min=0) * 50
            y = y * 50

            for k in range(y.shape[0]):
                mseH, _, maeH = calculate_evaluation_metrics(y[k][0], yHat[k][0])
                mseV, _, maeV = calculate_evaluation_metrics(y[k][1], yHat[k][1])
                MSE_H.append(mseH)
                MSE_V.append(mseV)
                MAE_H.append(maeH)
                MAE_V.append(maeV)
                gt.append(np.array([y[k][0], y[k][1]]))

    MSE = [np.average(MSE_H), np.average(MSE_V)]
    MAE = [np.average(MAE_H), np.average(MAE_V)]
    RMSE = [MSE[0] ** 0.5, MSE[1] ** 0.5]
    gt = np.array(gt)
    y_mean = np.mean(gt, 0)

    R_2 = [1 - (np.sum(MSE_H) / np.sum(np.square((y_mean - gt)[:, 0]))),
           1 - (np.sum(MSE_V) / np.sum(np.square((y_mean - gt)[:, 1])))]
    torch.cuda.empty_cache()

    return [MSE, MAE, RMSE, R_2]


def print_and_write(model, data, dataset, txt_path):
    print('------------' + model + '------------- ')
    print('--------|  MSE  |  MAE  |  RMSE  |  R_2  ]')
    print('| ' + dataset + '-H |{:.4f}|{:.4f}|{:.4f}|{:.5f}|'.format(data[0, 0], data[1, 0], data[2, 0],
                                                                     data[3, 0]))
    print('| ' + dataset + '-V |{:.4f}|{:.4f}|{:.4f}|{:.5f}|'.format(data[0, 1], data[1, 1], data[2, 1],
                                                                     data[3, 1]))
    with open(txt_path, 'a') as f:
        f.write('\n------------' + model + '------------- \n')
        f.write('--------|  MSE  |  MAE  |  RMSE  |  R_2  |\n')
        f.write('| ' + dataset + '-H |{:.4f}|{:.4f}|{:.4f}|{:.5f}|\n'.format(data[0, 0], data[1, 0], data[2, 0],
                                                                             data[3, 0]))
        f.write('| ' + dataset + '-V |{:.4f}|{:.4f}|{:.4f}|{:.5f}|\n\n'.format(data[0, 1], data[1, 1], data[2, 1],
                                                                               data[3, 1]))


def to_excel(model_name, data, output_path):
    print("output to excel")
    # [MSE,MAE,RMSE,R_2]
    out_data = {}
    eval_name_list = ['MSE', 'MAE', 'RMSE', 'R_2']
    dataset_name_list = ['TRA', 'TST', 'VAL']
    for i in range(len(dataset_name_list)):
        for k in range(len(eval_name_list)):
            out_data[dataset_name_list[i] + '-H-' + eval_name_list[k]] = data[i, :, k, 0]
            out_data[dataset_name_list[i] + '-V-' + eval_name_list[k]] = data[i, :, k, 1]

    if os.path.exists(output_path) is False:
        pd.DataFrame().to_excel(output_path)
    dataframe = pd.DataFrame(out_data)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        dataframe.to_excel(writer, sheet_name=model_name)


def val(ChooseModel, dir_path):
    model_name = get_model_name(ChooseModel)
    Folds_path = os.listdir(dir_path)
    Folds_path = [i for i in Folds_path if i[0] != '.']
    Folds_path.sort(key=lambda x: int(x.split('-')[-1]))
    TST = []
    VAL = []
    TRA = []
    for filename in Folds_path:
        full_path = os.path.join(dir_path, filename)
        if filename[0] == 'F' and args.start_fold <= int(full_path.split('-')[-1]) <= args.stop_fold:
            fold_path = os.listdir(full_path)
            for subfilename in fold_path:
                ##D:\。\Underground\sci\20-2-Fold-Dataset-1\Fold-9\...
                ffull_path = os.path.join(full_path, subfilename)
                if subfilename == model_name[:-1]:
                    writer = SummaryWriter('logs/test/' + subfilename[:-3] + '/' + filename)
                    print('Processing:' + subfilename[:-3] + '   ' + filename)
                    # [MSE,MAE,RMSE,R_2]
                    TST.append(
                        getData(ChooseModel, ffull_path + '/' + subfilename + '_best.pth', full_path + '/test.txt',
                                args.batch_size))
                    add_scalars(writer, TST, int(full_path.split('-')[-1]), 'TST')

                    TRA.append(
                        getData(ChooseModel, ffull_path + '/' + subfilename + '_best.pth', full_path + '/train.txt',
                                args.batch_size))
                    add_scalars(writer, TRA, int(full_path.split('-')[-1]), 'TRA')

                    VAL.append(
                        getData(ChooseModel, ffull_path + '/' + subfilename + '_best.pth',
                                'dataset/dataset_VAL_for_all.txt',
                                args.batch_size))
                    add_scalars(writer, VAL, int(full_path.split('-')[-1]), 'VAL')
    # [[[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2]...]
    # ,[[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2]...]
    # ,[[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2],[MSE,MAE,RMSE,R_2]...]]
    return np.array([TRA, TST, VAL])


def add_scalars(writer, data, fold, data_name):
    writer.add_scalar('MSE/' + data_name + '/H', data[-1][0][0], fold)
    writer.add_scalar('MAE/' + data_name + '/H', data[-1][1][0], fold)
    writer.add_scalar('RMSE/' + data_name + '/H', data[-1][2][0], fold)
    writer.add_scalar('R^2/' + data_name + '/H', data[-1][3][0], fold)
    writer.add_scalar('MSE/' + data_name + '/V', data[-1][0][1], fold)
    writer.add_scalar('MAE/' + data_name + '/V', data[-1][1][1], fold)
    writer.add_scalar('RMSE/' + data_name + '/V', data[-1][2][1], fold)
    writer.add_scalar('R^2/' + data_name + '/V', data[-1][3][1], fold)


def Print_Write(model_name, fold, data, output_path):
    '''

    Parameters
    ----------
    model_name
    fold fold number
    data  [TRA,TST,VAL]

    Returns
    -------

    '''
    print_and_write(model_name + ' FOLD' + fold, data[0], 'TRA', output_path)
    print_and_write(model_name + ' FOLD' + fold, data[1], 'TST', output_path)
    print_and_write(model_name + ' FOLD' + fold, data[2], 'VAL', output_path)


def Val(ChooseModel, dir_path, output_path, start_fold, stop_fold):
    data = val(ChooseModel, dir_path)  # [TRA,TST,VAL]
    print(data.shape)
    for i in range(stop_fold - start_fold + 1):
        Print_Write(get_model_name(ChooseModel), str(i + start_fold), data[:, i], output_path + '.txt')
    to_excel(get_model_name(ChooseModel), data, output_path + '.xlsx')
    Print_Write(get_model_name(ChooseModel), 'MEAN', [np.mean(data[0], 0), np.mean(data[1], 0), np.mean(data[2], 0)],
                output_path + '.txt')


'''
models to choose from
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
if __name__ == '__main__':

    start_fold = args.start_fold
    stop_fold = args.stop_fold

    output_path = args.output_path

    dir_path = args.dir_path
    for i in [9]:
        Val(i, dir_path, output_path, start_fold, stop_fold)

    # ---------------------------------------------------------------------
