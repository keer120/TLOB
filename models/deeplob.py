from torch import nn
import torch


class DeepLOB(nn.Module):
    def __init__(self):
        super().__init__()


        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(f"[DEBUG] input x shape: {x.shape}")
        x = x[:, None, :, :]  # [batch, 1, seq_len, features]
        print(f"[DEBUG] after unsqueeze x shape: {x.shape}")
        x = self.conv1(x)
        print(f"[DEBUG] after conv1 x shape: {x.shape}")
        x = self.conv2(x)
        print(f"[DEBUG] after conv2 x shape: {x.shape}")
        x = self.conv3(x)
        print(f"[DEBUG] after conv3 x shape: {x.shape}")

        x_inp1 = self.inp1(x)
        print(f"[DEBUG] x_inp1 shape: {x_inp1.shape}")
        x_inp2 = self.inp2(x)
        print(f"[DEBUG] x_inp2 shape: {x_inp2.shape}")
        x_inp3 = self.inp3(x)
        print(f"[DEBUG] x_inp3 shape: {x_inp3.shape}")

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        print(f"[DEBUG] x after concat shape: {x.shape}")
        x = x.permute(0, 2, 1, 3)  # [batch, seq_len, 192, ?]
        print(f"[DEBUG] x after permute shape: {x.shape}")
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)  # [batch, seq_len, features]
        print(f"[DEBUG] x before LSTM shape: {x.shape}")

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.softmax(out)
        return out