"""
LBF(M)モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Encoder
入力：潜在変数zとメッセージmのペア
出力：潜在信念変数betaの平均値と標準偏差
"""
class BetaEncoder(nn.Module):
    """ LBF encoder """
    def __init__(self, m_dim, z_dim, beta_dim):
        super(BetaEncoder, self).__init__()

        self.fc1 = nn.Linear(m_dim+z_dim, 1024)
        self.fc_mean = nn.Linear(1024, beta_dim)
        self.fc_logstd = nn.Linear(1024, beta_dim)

    def forward(self, m, z):
        # m : (B_SIZE, m_dim)
        # z : (B_SIZE, z_dim)
        x = torch.cat([m, z], dim=1) # (B_SIZE, m_zim+z_dim)
        x = F.relu(self.fc1(x))

        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)

        return mean, logstd


"""
Decoder
入力：潜在変数beta_t
出力：予測 [z_t, .., z_{t+pred_z_steps-1}]
"""
class ZSeqDecoder(nn.Module):
    """ LBF decoder """
    def __init__(self, beta_dim, z_dim, pred_z_steps):
        super(ZSeqDecoder, self).__init__()
        self.beta_dim = beta_dim
        self.z_dim = z_dim
        self.pred_z_steps = pred_z_steps

        self.fc1 = nn.Linear(beta_dim, 1024)
        self.fc2 = nn.Linear(1024, pred_z_steps*z_dim)

    def forward(self, beta):
        # (B_SIZE, beta_dim)
        x = F.relu(self.fc1(beta)) # (B_SIZE, 1024)
        x = self.fc2(x)   # (B_SIZE, pred_z_steps*z_dim)  最終層は活性化関数を適用しない
        pred_z_seq = x.view(x.size(0), self.pred_z_steps, self.z_dim) # (B_SIZE, pred_z_steps, z_dim)
        return pred_z_seq


class LBF(nn.Module):
    def __init__(self, beta_dim, m_dim, z_dim, pred_z_steps):
        super(LBF, self).__init__()
        self.encoder = BetaEncoder(m_dim, z_dim, beta_dim)
        self.decoder = ZSeqDecoder(beta_dim, z_dim, pred_z_steps)
      
    def sample_beta(self, mean, logstd):
        # 再パラメータ化トリック
        epsilon = torch.randn_like(mean)
        return mean + logstd.exp() * epsilon

    def forward(self, m, z):
        beta_mean, beta_logstd = self.encoder(m, z)
        beta = self.sample_beta(beta_mean, beta_logstd)

        pred_z_seq = self.decoder(beta)
        return beta, beta_mean, beta_logstd, pred_z_seq

    def loss(self, target_z_seq, pred_z_seq, beta_mean, beta_logstd):
        # target_z_seq, pred_z_seq : (B_SIZE, pred_z_steps, z_dim)
        # betaの分布と正規分布間のKLダイバージェンス
        KL_loss = -0.5 * torch.sum(1 + 2*beta_logstd - beta_mean**2 - (2*beta_logstd).exp()) / beta_mean.shape[0]

        # 予測誤差
        pred_loss = F.mse_loss(target_z_seq, pred_z_seq, reduction='sum') / (pred_z_seq.size(0) * pred_z_seq.size(1))
        return KL_loss+pred_loss, KL_loss, pred_loss
