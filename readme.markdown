
# ConvRnet 

This is the implementation of the ConvRnet model proposed in the paper, **Research on Underground 3-D Displacement Measurement Based on Convolutional Neural Networks and Dual Mutual Inductance Voltages**, for predicting 3D underground displacements.
![img.png](assert/img.png)
![img_2.png](assert/img_2.png)

## Getting Started 🚀

### Prerequisites 🛠️
It is recommended that you have an Nvidia GPU with **at least** 8GB of memory, as this will significantly reduce the time required for training and validation.

#### Software Requirements 🖥️

```
numpy~=1.24.1  
torch~=2.2.1+cu118  
torchvision~=0.17.1+cu118  
scipy~=1.10.1  
matplotlib~=3.7.5  
pandas~=2.0.3
```

### Installation 💻

1. Clone the repository
```bash
git clone https://github.com/ZHN202/ConvRnet.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage ℹ️

**Training 🏋️**
```
Models to choose from:
1 ---> Linear MLP  
2 ---> Conv1d  
3 ---> ConvRnet  
4 ---> ConvRnet_linear  
5 ---> ConvRnet_without_CBAM  
6 ---> ConvRnet_without_DM  
7 ---> ConvMLP  
8 ---> RBF  
9 ---> RBF_MLP  
```
```bash
python train_for_k_fold.py --ChooseModel=1
```

**Validation ✔️**
```bash
python val_to_file.py --dir_path=your/path/to/20-4-Fold-Dataset-1
```

## Citations 📚
If you use this code in your research, please cite:
```
@article{jia2024research,
  title={Research on Underground 3-D Displacement Measurement Based on Convolutional Neural Networks and Dual Mutual Inductance Voltages},
  author={Jia, Shengyao and Zhou, Haonan and Shi, Ge and Chen, Haiwei and Han, Jianqiang and Li, Qing},
  journal={IEEE Sensors Journal},
  volume={24},
  number={1},
  pages={526--532},
  year={2024}
}
```

