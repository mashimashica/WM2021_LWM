"""
Speakerモデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    # if has_torch_function_unary(logits):
    #     return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    # if eps != 1e-10:
    #     warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft

    return y_soft, y_hard

"""
Encoder
入力：画像x
出力：メッセージm
"""
class Encoder(nn.Module):
    """ Message encoder """
    def __init__(self, m_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)

        #順伝播
        self.fc = nn.Linear(2*2*128, m_dim)
        

    def forward(self, x):
        # x : [B_SIZE, 3, 32, 32]
        x = F.relu(self.conv1(x)) # [B_SIZE, 32, 15, 15]
        x = F.relu(self.conv2(x)) # [B_SIZE, 64, 6, 6] 
        x = F.relu(self.conv3(x)) # [B_SIZE, 128, 2, 2]  
        x = x.view(x.size(0), -1) # [B_SIZE, 512]
        
        x = self.fc(x)
        p_soft, m = gumbel_softmax(x, tau=10, hard=False)

        return p_soft, m


"""
Decoder
入力：メッセージm
出力：再構成画像recon_x
"""
class Decoder(nn.Module):
    """ Message decoder """
    def __init__(self, m_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(m_dim, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
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
