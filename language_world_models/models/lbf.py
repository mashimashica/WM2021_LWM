"""
LBF(M)モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Encoder
入力：潜在変数zとメッセージmのペア
出力：潜在変数zの平均値と標準偏差
"""
class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim

        self.fc1 = nn.Linear(m_dim + z_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.fc_mean = nn.Linear(1024, beta_dim)
        self.fc_logstd = nn.Linear(1024, beta_dim)

    def forward(self, m, z):
        x = F.relu(self.fc1(torch.cat([m, z], dim=1))
        x = F.relu(self.fc2(x)) 

        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)

        return mean, logstd


"""
Decoder
入力：潜在変数beta_t
出力：予測 [z_t, .., z_t+D]
"""
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, beta_dim):
        super(Decoder, self).__init__()
        self.beta_dim = beta_dim

        self.fc1 = nn.Linear(beta_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, z_dim)

    def forward(self, beta):
        x = F.relu(self.fc1(beta))
        x = F.relu(self.fc2(x)) 
        prediction = self.fc3(x)
        return prediction


class LBF(nn.Module):
    def __init__(self, beta_dim, m_dim, z_dim, pred_z_steps):
        super(LBF, self).__init__()
        self.encoder = Encoder(beta_dim, m_dim, z_dim)
        self.decoder = Decoder(beta_dim, z_dim, pred_z_steps)
      
    def sample_beta(self, mean, logstd):
        # 再パラメータ化トリック
        epsilon = torch.randn_like(mean)
        return mean + logstd.exp() * epsilon

    def forward(self, m, z):
        beta_mean, beta_logstd = self.encoder(m, z)
        beta = self.sample_beta(beta_mean, beta_logstd)

        recon_x = self.decoder(z)
        return pred_zs, beta, beta_mean, beta_logstd

    def loss(self, correct_zs, predict_zs, beta_mean, beta_logstd):
        # KLダイバージェンス
        KL_loss = -0.5 * torch.sum(1 + 2*beta_logstd - beta_mean**2 - (2*beta_logstd).exp()) / beta_mean.shape[0]

        # 予測誤差
        pred_loss = F.mse_loss(correct_zs, predict_zs, reduction='sum') / beta_mean.shape[0]
        return KL_loss+pred_loss, KL_loss, pred_loss
