import torch
import torch.nn as nn
from torchvision import models

from models.CBAM import cbam_block


class ConvRnet(nn.Module):
    def __init__(self):
        super(ConvRnet, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------

        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        """
            x1-->2 data map   [2,51,51]
            x2-->3 sensor      [1,3]
        """
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)

        # upscale sensor value
        conv1_out = self.conv1(x2)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)
        x1 = torch.cat((x1, conv1_out), 1)

        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)

        # Feature Extraction Layer Input:[3, 51, 51]
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)

        # Output：256*13*13
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class ConvMLP(nn.Module):
    def __init__(self):
        super(ConvMLP, self).__init__()
        self.pic = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 86, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(86),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------

        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(514, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
            x1-->2 data map   [2,51,51]
            x2-->3 sensor      [1,3]
        """
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)

        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)

        # Feature Extraction Layer Input:[3, 51, 51]
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)

        # Output：256*13*13
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)

        # upscale sensor value
        conv1_out = self.conv1(x2)

        conv1_out = conv1_out.view(-1, 3 * 86)
        x1 = torch.cat((x1, conv1_out), 1)

        out = self.out(x1)
        return out


class ConvRnet_linear(nn.Module):
    def __init__(self):
        super(ConvRnet_linear, self).__init__()
        # self.pic = None

        self.conv1 = nn.Sequential(
            nn.Linear(3, 2601),
            nn.BatchNorm1d(2601),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1]
        conv1_out = self.conv1(x2)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)
        x1 = torch.cat((x1, conv1_out), 1)
        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class ConvRnet_without_CBAM(nn.Module):
    def __init__(self):
        super(ConvRnet_without_CBAM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)
        conv1_out = self.conv1(x2)
        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)
        x1 = torch.cat((x1, conv1_out), 1)
        x1 = self.conv2d_1(x1)
        x1 = self.residual_block_1(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out


class ConvRnet_without_DM(nn.Module):
    def __init__(self):
        super(ConvRnet_without_DM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.LeakyReLU(0.1)
        )
        # 3->64  51->51 --------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam1 = cbam_block(64, ratio=4, kernel_size=7)
        # --------------------------------------------------

        # 64->64  51->51 --------------------------------------------------
        self.residual_block_1 = nn.Sequential(*list(models.resnet18().children())[4:5])
        self.cbam2 = cbam_block(64, ratio=4, kernel_size=3)
        # -----------------------------------------------------------------

        # 64->128  51->26 ---------------------------------------------------
        self.residual_block_2 = nn.Sequential(*list(models.resnet18().children())[5:6])
        self.cbam3 = cbam_block(128, ratio=4, kernel_size=3)
        # ------------------------------------------------------------

        # 128->256  26->13 ---------------------------------------------------------
        self.residual_block_3 = nn.Sequential(*list(models.resnet18().children())[6:7])  # bc*256*4*4
        self.cbam4 = cbam_block(256, ratio=4, kernel_size=3)
        # ---------------------------------------------------------------------------
        self.avgPooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)

        conv1_out = self.conv1(x2)

        conv1_out = conv1_out.view(-1, 1, 2601).view(-1, 1, 51, 51)

        x1 = conv1_out
        x1 = self.conv2d_1(x1)
        x1 = self.cbam1(x1)
        x1 = self.residual_block_1(x1)
        x1 = self.cbam2(x1)
        x1 = self.residual_block_2(x1)
        x1 = self.cbam3(x1)
        x1 = self.residual_block_3(x1)
        x1 = self.cbam4(x1)
        x1 = self.avgPooling(x1)
        x1 = x1.view(-1, 256 * 1 * 1)
        out = self.out(x1)
        return out
