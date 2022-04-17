import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.utils as vutils

# TODO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_speaker(args, model_speaker, i_episode, x):
    model_speaker.eval()
    _, _, recon_x = model_speaker(x)

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
def test_vae(args, model_vae, i_episode, x):
    model_vae.eval()
    _, _, _, recon_x = model_vae(x)

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
    #vutils.save_image(recon_x.to(device)[:16], "vae_recon.png", normalize=True, nrow=4)
    
    result_vae_path = os.path.join(args.results_dir, "train_vae_%06d.png" % (i_episode+1))
    fig.savefig(result_vae_path)

    plt.close(fig)
