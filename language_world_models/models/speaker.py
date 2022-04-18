"""
Speakerモデル
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
    def __init__(self, m_dim):
        super(Encoder, self).__init__()
        self.m_dim = m_dim

        self.fc1 = nn.Linear(3*11*11, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, m_dim)

    def forward(self, x):
        # x : [B_SIZE, 3, 11, 11]
        x = x.view(x.size(0), -1) # [B_SIZE, 363]
        x = F.relu(self.fc1(x)) # [B_SIZE, 1024]
        x = F.relu(self.fc2(x)) # [B_SIZE, 1024]
        x = self.fc3(x) # [B_SIZE, m_dim]  

        # self.p_soft, self.m = gumbel_softmax(x, tau=0.1)
        p_soft = F.gumbel_softmax(x, tau=0.001, hard=False)
        dim = -1
        index = p_soft.max(dim, keepdim=True)[1]
        p_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        m = p_hard - p_soft.detach() + p_soft

        return p_soft, m


"""
Decoder
入力：潜在変数z
出力：再構成画像recon_x
"""
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, m_dim):
        super(Decoder, self).__init__()
        self.m_dim = m_dim

        self.fc1 = nn.Linear(m_dim, 1024)
        self.fc2 = nn.Linear(1024, 3*11*11)

    def forward(self, x):
        # x : [B_SIZE, m_dim]
        x = F.relu(self.fc1(x)) # [B_SIZE, 1024]
        x = torch.sigmoid(self.fc2(x)) # [B_SIZE, 3*11*11]
        recon = x.view(x.size(0), 3, 11, 11) # [B_SIZE, 3, 11, 11]
        return recon

# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

class Speaker(nn.Module):
    def __init__(self, m_dim):
        super(Speaker, self).__init__()
        self.encoder = Encoder(m_dim)
        self.decoder = Decoder(m_dim)

    def forward(self, x):
        p_soft, m = self.encoder(x)

        recon_x = self.decoder(m)
        return m, p_soft, recon_x


    def loss(self, x, recon_x, p_soft):
        # p_soft : (B_SIZE, m_dim)
        # エントロピー（メッセージが与える情報量）
        p_soft_mean = p_soft.mean(dim=0) # (m_dim)
        n_entropy = (p_soft_mean * torch_log(p_soft_mean)).sum()

        # 再構成誤差
        MSE = F.mse_loss(recon_x, x, reduction='sum')/x.shape[0]

        CC = n_entropy + MSE
        return CC, n_entropy, MSE
