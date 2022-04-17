import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils

from envs import create_ChoosePathGridDefaultEnv
from models.speaker import Speaker
from utils import misc, option


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_speaker(args, model_speaker, i_episode, x):
    model_speaker.eval()
    recon_x, _ = model_speaker(x)

    # 入力画像の表示
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Input")
    plt.imshow(np.transpose(vutils.make_grid(x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))

    # 再構成画像の表示
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Reconstruction")
    plt.imshow(np.transpose(vutils.make_grid(recon_x.to(device)[:16], normalize=True, nrow=4).cpu(),(1,2,0)))
    
    result_vae_path = os.path.join(args.results_dir, "train_speaker_%06d.png" % (i_episode+1))
    fig.savefig(result_vae_path)

    plt.close(fig)

@torch.no_grad()
def test(args, env, model_speaker, i_episode):
    obs_speaker_list = misc.get_many_head_frame(env, 16)
     
    # Speaker のテスト
    x = torch.stack(obs_speaker_list).to(device)
    test_speaker(args, model_speaker, i_episode, x)


# Speaker の更新
def train_batch_speaker(model_speaker, optimizer_speaker, x):
    model_speaker.train()
    model_speaker.zero_grad()

    recon_x, p_soft = model_speaker(x)
    loss, n_entropy, recon_loss = model_speaker.loss(x, recon_x, p_soft)
    loss.backward()
    optimizer_speaker.step()
    
    return loss.item(), n_entropy.item(), recon_loss.item()

def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    losses = {'speaker' : {'loss':[], 'n_entropy':[], 'recon_loss':[]}} 

    # ゲーム環境のインスタンスの取得
    env = create_ChoosePathGridDefaultEnv()

    # Speakerの定義
    model_speaker = Speaker(args.m_dim).to(device)
    print(model_speaker)
    optimizer_speaker = optim.Adam(model_speaker.parameters())

    test(args, env, model_speaker, -1)

    
    obs_speaker_list = misc.get_many_head_frame(env, 10240)
    train_speaker_dataset = torch.utils.data.TensorDataset(torch.stack(obs_speaker_list))
    
    dataloader = torch.utils.data.DataLoader(
        train_speaker_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2, 
        drop_last=True,
        pin_memory=True
    )

    last_time = time.time()

    # 訓練ループ
    for i_epoch in range(args.num_epochs):
        
        # Speaker の更新
        for batch in dataloader:
            batch = batch[0].to(device)
            loss, n_entropy, recon_loss = train_batch_speaker(model_speaker, optimizer_speaker, batch)
            losses['speaker']['loss'].append(loss)
            losses['speaker']['n_entropy'].append(n_entropy)
            losses['speaker']['recon_loss'].append(recon_loss)
  
        
        if (i_epoch+1) % args.print_freq == 0:
            print('epoch: %d / %d (%d sec)' % (i_epoch+1, args.num_epochs, time.time()-last_time))
            last_time = time.time()
             
            # Speaker について表示
            print('\tSpeaker : Loss: %lf  (Negative entropy : %lf,  Reconstruction loss : %lf)' %
                  (np.average(losses['speaker']['loss']),  np.average(losses['speaker']['n_entropy']), np.average(losses['speaker']['recon_loss'])))
            losses['speaker']['loss'].clear()
            losses['speaker']['n_entropy'].clear()
            losses['speaker']['recon_loss'].clear()

        if (i_epoch+1) % args.save_freq == 0:
            model_speaker_path = os.path.join(args.checkpoints_dir, "model_speaker_%06d.pth" % (i_epoch+1))
            torch.save(model_speaker.state_dict(), model_speaker_path)

        if (i_epoch+1) % args.test_freq == 0:
            test(args, env, model_speaker, i_epoch)


if __name__ == "__main__":
    args = option.parse_args()
    train(args)
