"""
VAE(V)モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Encoder
入力：画像x
出力：潜在変数zの平均値と標準偏差
"""
class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)

        self.fc_mean = nn.Linear(2*2*128, z_dim)
        self.fc_logstd = nn.Linear(2*2*128, z_dim)

    def forward(self, x):
        # x : [B_SIZE, 3, 32, 32]
        x = F.relu(self.conv1(x)) # [B_SIZE, 32, 15, 15]
        x = F.relu(self.conv2(x)) # [B_SIZE, 64, 6, 6] 
        x = F.relu(self.conv3(x)) # [B_SIZE, 128, 2, 2]  
        x = x.view(x.size(0), -1) # [B_SIZE, 512]

        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)

        return mean, logstd


"""
Decoder
入力：潜在変数z
出力：再構成画像recon_x
"""
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 6, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 6, stride=2)

    def forward(self, x):
        # x : [B_SIZE, z_dim]
        x = F.relu(self.fc1(x)) # [B_SIZE, 512]
        x = x.unsqueeze(-1).unsqueeze(-1) # [B_SIZE, 512, 1, 1]
        x = F.relu(self.deconv1(x)) # [B_SIZE, 128, 5, 5]
        x = F.relu(self.deconv2(x)) # [B_SIZE, 64, 14, 14]
        recon = F.sigmoid(self.deconv3(x)) # [B_SIZE, 3, 32, 32]
        return recon


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
      
    def sample_z(self, mean, logstd):
        # 再パラメータ化トリック
        epsilon = torch.randn_like(mean)
        return mean + logstd.exp() * epsilon

    def forward(self, x):
        z_mean, z_logstd = self.encoder(x)
        z = self.sample_z(z_mean, z_logstd)

        recon_x = self.decoder(z)
        return z, z_mean, z_logstd, recon_x

    def loss(self, x, recon_x, z_mean, z_logstd):
        # KLダイバージェンス
        KL_loss = -0.5 * torch.sum(1 + 2*z_logstd - z_mean**2 - (2*z_logstd).exp()) / x.shape[0]

        # 再構成誤差
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
        # recon_loss = F.binary_cross_entropy(recon_x, x)
        return KL_loss+recon_loss, KL_loss, recon_loss
