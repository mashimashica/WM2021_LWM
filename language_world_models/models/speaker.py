"""
VAE(V)モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -1) -> torch.Tensor:
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
出力：潜在変数zの平均値と標準偏差
"""
class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, m_dim):
        super(Encoder, self).__init__()
        self.m_dim = m_dim

        self.fc1 = nn.Linear(3*11*11, 1024)
        self.fc2 = nn.Linear(1024, m_dim)

    def forward(self, x):
        # x : [B_SIZE, 3, 11, 11]
        x = x.view(x.size(0), -1) # [B_SIZE, 363]
        x = F.relu(self.fc1(x)) # [B_SIZE, 1024]
        x = F.relu(self.fc2(x)) # [B_SIZE, m_dim]  
        p_soft, m = gumbel_softmax(x, tau=1)
        p_soft = x
        m = x

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
        self.p_soft = p_soft 
        self.m = m

        recon_x = self.decoder(m)
        return m, p_soft, recon_x


    def loss(self, x, recon_x, p_soft):
        #入力サイズ p_soft : [B_SIZE, m_dim]
        # 再構成誤差
        MSE = F.mse_loss(recon_x, x, reduction='sum')/x.shape[0]
        p_soft_mean = p_soft.mean(dim=0) # (m_dim)
        # print(p_soft.shape)
        # print(p_soft_mean.shape)
        n_entropy = (p_soft_mean * torch_log(p_soft_mean)).sum()
        CC = n_entropy + MSE
        # print(CC)
        # recon_loss = F.binary_cross_entropy(recon_x, x)
        return CC, n_entropy, MSE
