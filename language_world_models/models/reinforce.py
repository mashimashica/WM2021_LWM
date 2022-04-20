"""
REINFROCE(C)モデル
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F


def discount_rewards(rews, gamma, n_steps=None):
    # Takes rewards in a batch, and calculates discounted returns
    # Note that returns are truncated at the end of the batch
    # nstep controls how many steps you take the sum over
    rews_disc = np.zeros((len(rews),))
    rews_temp = np.zeros((len(rews),))
    for i in range(len(rews) - 1, -1, -1):
        rews_temp[i] = rews[i]
        rews_temp[i+1:] *= gamma
        if n_steps is None:
            rews_disc[i] = sum(rews_temp)
        else:
            rews_disc[i] = sum(rews_temp[i: i + n_steps])
    if n_steps is None:
        return rews_disc
    return rews_disc[:len(rews)]


class REINFORCE(nn.Module):
    """
    An agent that uses policy gradient (REINFORCE)
    Uses a 2-layer MLP policy.
    """
    def __init__(self, z_dim, beta_dim, act_dim=4, hid_dim=1024, gamma=0.9, ent_coeff=1e-2, val_coeff=1e-1, n_steps=None):
        """Sets variables, and creates layers for networks

        Args:
            in_dim:      input/ observation size
            hid_dim:      size of the hidden layers
            act_dim:      output size (for actions)
            gamma:      discount factor
            ent_coeff:  coefficient for entropy bonus
            val_coeff:  coefficient for value learning
            n_steps:    number of steps used to calculate discounted reward.
                        if None, only the next reward is given
        """

        super(REINFORCE, self).__init__()

        # Main network
        self.h1 = nn.Linear(z_dim+beta_dim, hid_dim)
        self.h2 = nn.Linear(hid_dim, hid_dim)
        self.out_a = nn.Linear(hid_dim, act_dim)
        self.v = nn.Linear(hid_dim, 1)

        # Other properties and coefficients
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.val_coeff = val_coeff
        self.n_steps = n_steps

    def forward(self, z, beta):
        # Computes a forward pass through the network.
        # Main network, used to compute action a and value v
        x = torch.cat([z, beta], dim=-1)
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        a = self.out_a(x)
        v = self.v(x)

        return a, v

    def act(self, z, beta):
        # Returns a discrete sample from the main network
        # Output is an action
        logits_a, v = self.forward(z, beta)
        return dists.Categorical(logits=logits_a).sample()

    def loss(self, acts, rewards, z, beta):
        # Updates main network
        # comms, acts are the lists of messages / actions actually taken

        # Calculated discounted reward
        rews_disc = torch.tensor(discount_rewards(rewards, self.gamma, self.n_steps)).to(acts.device)

        # Pass observations through main network, obtain probability distributions over actions/ messages
        logits_a, V = self.forward(z, beta)
        probs_a = F.softmax(logits_a, dim=1)

        # Convert to log probabilities, index by actions/ messages taken
        logprobs_a = dists.Categorical(probs=probs_a).log_prob(acts)

        # Clamp probabilities to avoid NaNs when computing entropy bonus
        probs_a = torch.clamp(probs_a, 1e-6, 1)

        # Calculate losses using policy gradients
        loss_a = (-logprobs_a * (rews_disc - V)).mean()
        # Calculate entropy bonuses
        loss_ent_a = (-probs_a * torch.log(probs_a)).sum(dim=1).mean()
        # Calculate value function losses (MSE)
        loss_v = ((rews_disc - V) ** 2).mean()

        # Total loss is the weighted sum of all previous losses.
        loss = loss_a - self.ent_coeff * loss_ent_a + self.val_coeff * loss_v

        return loss, loss_a, -loss_ent_a, loss_v
