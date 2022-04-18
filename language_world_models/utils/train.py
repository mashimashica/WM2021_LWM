from utils import misc

# Speaker の更新
def train_speaker_batch(model_speaker, optimizer_speaker, x):
    model_speaker.train()
    model_speaker.zero_grad()

    _, p_soft, recon_x = model_speaker(x)
    loss, n_entropy, recon_loss = model_speaker.loss(x, recon_x, p_soft)
    loss.backward()
    # misc.plot_grad_flow(model_speaker.named_parameters())
    optimizer_speaker.step()
    
    return loss.item(), n_entropy.item(), recon_loss.item()

# VAE の更新
def train_vae_batch(model_vae, optimizer_vae, x):
    model_vae.train()
    model_vae.zero_grad()

    _, z_mean, z_logstd, recon_x = model_vae(x)
    vae_loss, kl_loss, recon_loss = model_vae.loss(x, recon_x, z_mean, z_logstd)
    vae_loss.backward()
    optimizer_vae.step()
    
    return vae_loss.item(), kl_loss.item(), recon_loss.item()


"""
LBF の更新
input_m_seqs    : (SEQ_NUM, SEQ_LEN, m_dim)
input_z_seqs    : (SEQ_NUM, SEQ_LEN, z_dim)
target_z_seqs   : (SEQ_NUM, SEQ_LEN, pred_z_steps, z_dim)
pred_z_seqs     : (SEQ_NUM, SEQ_LEN, pred_z_steps, z_dim)
"""
def train_lbf_batch(model_lbf, optimizer_lbf, \
                    input_m_seqs, input_z_seqs, target_z_seqs):
    model_lbf.train()
    model_lbf.zero_grad()
     
    pred_z_seqs, beta, beta_mean, beta_logstd = \
            model_lbf(input_m_seqs, input_z_seqs)

    loss, kl_loss, pred_loss = \
            model_lbf.loss(target_z_seqs, pred_z_seqs, beta_mean, beta_logstd)
    loss.backward()
    # misc.plot_grad_flow(model_lbf.named_parameters())
    optimizer_lbf.step()
    return loss.item(), kl_loss.item(), pred_loss.item()

# REINFORCE の更新
def train_reinforce_batch(model_reinforce, optimizer_reinforce, x):
    model_reinforce.train()
    model_reinforce.zero_grad()

    _, z_mean, z_logstd, recon_x = model_vae(x)
    loss, kl_loss, recon_loss = model_vae.loss(x, recon_x, z_mean, z_logstd)
    loss.backward()
    optimizer_reinforce.step()
    
    return loss.item(), kl_loss.item(), recon_loss.item()
