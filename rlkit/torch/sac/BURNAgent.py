import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2



class BURNAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 high_level_policy,
                 low_level_policy,
                 c,
                 epsilon=01e-02,
                 **kwargs
                 ):
        super().__init__()
        self.use_goals=True
        self.latent_dim=latent_dim

        self.context_encoder=context_encoder
        self.high_level_policy=high_level_policy
        self.low_level_policy=low_level_policy

        self.recurrent=kwargs['recurrent']
        self.use_ib=kwargs['use_information_bottleneck']
        self.sparse_rewards=kwargs['sparse_rewards']
        self.use_next_obs_in_context=kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.c=c  #Horizon for low-level rollouts
        self.epsilon=epsilon #threshold for goal completion
        self.goal=None
        self.steps=0,
        #NOTE: This means that when we start a new trajectory we need to set that agent's steps to 0
        self.clear_z()
        self.clear_goal()

    def clear_z(self,num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)
    def clear_goal(self):
        self.goal=None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    #TODO double check
    def sample_goal(self,state,deterministic=False):
        state=ptu.from_numpy(state[None])
        z=self.z
        # z=torch.tensor(self.z).view(1,-1)
        #TUNE: what's going on here?
        in_=torch.cat((state,z),dim=1)
        self.goal,_= self.high_level_policy.get_action(in_,deterministic=deterministic)
        self.goal=np.array(self.goal)
        return self.goal

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        assert(self.goal is not None)
        goal=ptu.from_numpy(self.goal[None])
        goal=goal.detach()
        obs = ptu.from_numpy(obs[None])
        in_=torch.cat((obs,goal),dim=1)
        return self.low_level_policy.get_action(in_, deterministic=deterministic)

    def forward_low(self,obs_goals):
        #obs_goals is the concatenation of the observations and the goals
        policy_outputs = self.low_level_policy(obs_goals.detach(), reparameterize=True, return_log_prob=True)
        return policy_outputs

    def forward_high(self,obs,context):
        pass

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.high_level_policy,self.low_level_policy]
