from models.conv_1 import ConvRnet, ConvRnet_linear, ConvRnet_without_CBAM, ConvRnet_without_DM, ConvMLP
from models.otherModel import *
from models.rbf import *
from models.rbf_mlp import *
from models.rbf_utils import *


class argparse():
    batch_size = 32
    epochs = 200
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_seed(seed=42):
    import random
    # seed = 42  # The answer to the ultimate question of life, the universe, and everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class myLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.maeloss = myMAELoss(alpha=1)
        self.alpha = alpha

    def forward(self, input, target):
        theta = torch.abs(torch.abs(input - target)[:, 0] - torch.abs(input - target)[:, 1])

        return self.maeloss(input, target) + torch.mean(theta) * self.alpha


class myMSELoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss1 = nn.MSELoss(reduction='mean')
        self.loss2 = nn.MSELoss(reduction='mean')

    def forward(self, input, target):
        return torch.sum(self.loss1(input[:, 0], target[:, 0]) * self.alpha + self.loss2(input[:, 1], target[:, 1]))


class myMAELoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss1 = nn.L1Loss(reduction='mean')
        self.loss2 = nn.L1Loss(reduction='mean')

    def forward(self, input, target):
        return torch.sum(self.loss1(input[:, 0], target[:, 0]) * self.alpha + self.loss2(input[:, 1], target[:, 1]))


class MASELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSELoss = myMSELoss()
        self.MAELoss = myMAELoss()

    def forward(self, input, target):
        return self.MSELoss(input, target) + self.MAELoss(input, target)


def get_model(ChooseModel, device):
    if ChooseModel == 1:
        model = Linear().to(device)
    elif ChooseModel == 2:
        model = Conv1d().to(device)
    elif ChooseModel == 3:
        model = ConvRnet().to(device)
    elif ChooseModel == 4:
        model = ConvRnet_linear().to(device)
    elif ChooseModel == 5:
        model = ConvRnet_without_CBAM().to(device)
    elif ChooseModel == 6:
        model = ConvRnet_without_DM().to(device)
    elif ChooseModel == 7:
        model = ConvMLP().to(device)
    elif ChooseModel == 8:
        model = RBF(in_features_dim=3,
                    num_kernels=1024,
                    out_features_dim=2,
                    radial_function=rbf_inverse_multiquadric,
                    norm_function=euclidean_norm,
                    normalization=False).to(device)
    elif ChooseModel == 9:
        model = RBF_MLP(in_features_dim=3,
                        num_kernels=1024,
                        out_features_dim=64,
                        radial_function=rbf_inverse_multiquadric,
                        norm_function=euclidean_norm,
                        normalization=False).to(device)
    else:
        print('please_choose_the_right_model')
    return model


def get_model_name(ChooseModel):
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
    elif ChooseModel == 8:
        save_model_name = 'RBF_MS_'
    elif ChooseModel == 9:
        save_model_name = 'RBF_MLP_MS_'
    else:
        print('please_choose_the_right_model')

    return save_model_name


def load_weights(model, f):
    print('loading weights')
    weight = torch.load(f)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in weight.items(
    ) if k in model_state_dict.keys()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print('finish')


def draw_loss(train_epochs_loss, valid_epochs_loss, fig_name='output/graph/valid/loss.png'):
    plt.figure(figsize=(8, 8))

    # plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("each_epochs_loss")

    plt.legend()

    plt.savefig(fig_name)


# 评估图


def draw_valid(MSE_val, RMSE_val, MAE_val, fig_name='output/graph/valid/valid.png'):
    plt.figure(figsize=(16, 16))
    # Horizontal displacement   vertical displacement
    MSE_val = np.array(MSE_val)
    RMSE_val = np.array(RMSE_val)
    MAE_val = np.array(MAE_val)

    plt.subplot(221)
    plt.ylabel('MSE')
    plt.xlabel('Valid data no.')
    plt.plot(MSE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(MSE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("MSE")
    plt.legend()

    plt.subplot(222)
    plt.ylabel('RMSE')
    plt.xlabel('Valid data no.')
    plt.plot(RMSE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(RMSE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("RMSE")
    plt.legend()

    plt.subplot(223)
    plt.ylabel('MAE')
    plt.xlabel('Valid data no.')
    plt.plot(MAE_val[:, 0], '-o', label='Horizontal displacement')
    plt.plot(MAE_val[:, 1], '-o', label='Vertical displacement')
    plt.title("MAE")
    plt.legend()

    plt.subplot(224)
    data = np.array([np.average(MAE_val), np.average(MSE_val), np.sqrt(np.average(np.square(MAE_val)))])
    labels = ['MAE', 'MSE', 'RMSE']
    plt.bar(labels, data, 0.5)

    plt.title("Average")

    plt.savefig(fig_name)


# learning_rate_graph
def draw_lr(lr, fig_name='output/graph/train/learning_rate.png'):
    plt.figure(figsize=(8, 8))
    lr = np.array(lr)

    plt.ylabel('Learnisng rate')
    plt.xlabel('Epoch')
    plt.plot(lr, '-o')

    plt.savefig(fig_name)


# test_error_plot
def draw_error(gt, pred, fig_name='output/graph/train/error.png'):
    plt.figure(figsize=(16, 8))
    gt = np.array(gt)
    pred = np.array(pred)
    errors_h = gt[:, 0] - pred[:, 0]
    errors_v = gt[:, 1] - pred[:, 1]
    plt.subplot(221)
    plt.ylabel('displacement [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(gt[:, 0], '-o', label='Ground truth')
    plt.plot(pred[:, 0], '-o', label='Predition')
    plt.title("Horizontal displacement")
    plt.legend()

    plt.subplot(222)
    plt.ylabel('displacement [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(gt[:, 1], '-o', label='Ground truth')
    plt.plot(pred[:, 1], '-o', label='Predition')
    plt.title("Vertical displacement")

    plt.legend()
    plt.subplot(223)
    plt.ylabel('error [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(errors_h, '-o', label='Ground truth')
    plt.axhline(y=1, color='red')
    plt.axhline(y=-1, color='red')
    plt.title("Vertical displacement")

    plt.legend()
    plt.subplot(224)
    plt.ylabel('error [mm]')
    plt.xlabel('Valid data no.')
    plt.plot(errors_v, '-o', label='Ground truth')
    plt.axhline(y=1, color='red')
    plt.axhline(y=-1, color='red')
    plt.title("Vertical displacement")

    plt.legend()
    plt.savefig(fig_name)
    # plt.show()


def draw_pic(pics, fig_name='data_graph.png'):
    plt.figure(figsize=(12, 4))
    title = 'CNN input (Linear input:' + str(pics[-1][0]) + ' ' + str(pics[-1][1]) + ' ' + str(pics[-1][2]) + ')'
    plt.suptitle(title)

    plt.subplot(133)
    im1 = plt.imshow(pics[0])
    plt.colorbar(im1, fraction=0.05, pad=0.05)
    plt.title("Linear layer output")

    plt.subplot(131)
    im2 = plt.imshow(pics[1])
    plt.colorbar(im2, fraction=0.05, pad=0.05)
    plt.title("data graph 1")

    plt.subplot(132)
    im3 = plt.imshow(pics[2])
    plt.colorbar(im3, fraction=0.05, pad=0.05)
    plt.title("data graph 2")

    plt.savefig(fig_name)


def calculate_evaluation_metrics(gt, pred):
    mse = (gt - pred) ** 2
    rmse = mse ** 0.5
    mae = np.abs(gt - pred)
    return mse, rmse, mae
