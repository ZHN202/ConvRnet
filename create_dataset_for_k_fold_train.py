import os
import re

from utils.utils import *


def getData(file_path):
    with open(file_path,'r') as f:
        context=f.readlines()
    val=50
    train=0
    train_=0
    for line in context:
        if 'Train' in line:
            train_=float(re.findall(r'\d+\.\d+', line)[0])
        if 'Val' in line:
            if val>float(re.findall(r'\d+\.\d+', line)[0]):
                val=float(re.findall(r'\d+\.\d+', line)[0])
                train=train_
    return [train,val]


def print_and_write(model_name,data,txt_path):

    with open(txt_path, 'a') as f:
        print('------------' + model_name + '------------- ')
        f.write('\n------------' + model_name + '------------- \n')
        for i in data:
            print('--------FOLD-'+i[0]+'----------')
            f.write('--------FOLD-' + i[0] + '----------')
            print('Train Loss:'+str(i[1][0])+'    Val Loss:'+str(i[1][1]))
            f.write('Train Loss:'+str(i[1][0])+'    Val Loss:'+str(i[1][1])+'\n')


def Write(dir_path,ChooseModel):
    model_name=get_model_name(ChooseModel)
    Folds_path = os.listdir(dir_path)
    data=[]
    for filename in Folds_path:
        full_path = os.path.join(dir_path, filename)
        fold_path = os.listdir(full_path)
        for subfilename in fold_path:
            print('PROCESSING' + filename)
            ffull_path = os.path.join(full_path, subfilename)
            if subfilename == model_name[:-1]:
                data.append((filename,
                    getData(ffull_path + '/log/' + subfilename + '_log.txt')))
    print_and_write(model_name[:-1],data,'output2.txt')

if __name__ == '__main__':

    dir_path = r'D:\。\Underground\sci\20-4-Fold-Dataset-1'
    for i in range(1,10):
        Write(dir_path,i)




