from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm



class BURNSoftActorCritic(MetaRLAlgorithm):
    def __init__(self,
                 env,
                 train_tasks,
                 eval_tasks,
                 latent_dim,
                 nets,#=[agent,q_list,vf_list]

                 h_policy_lr=1e-3,
                 l_policy_lr=1e-3,
                 h_qf_lr=1e-3,
                 l_qf_lr=1e-3,
                 h_vf_lr=1e-3,
                 l_vf_lr=1e-3,
                 context_lr=1e-3,

                 kl_lambda=1.,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0,
                 optimizer_class=optim.Adam,
                 recurrent=False,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,
                 sparse_rewards=False,

                 soft_target_tau=1e-2,
                 plotter=None,
                 render_eval_paths=False,


                 **kwargs
                 ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            use_goals=True,
            **kwargs
        )


        self.low_Q1,self.low_Q2=nets[1][0]
        self.high_Q1, self.high_Q2 = nets[1][1]
        self.low_vf,self.high_vf=nets[2]

        self.low_vf_target,self.high_vf_target=self.low_vf.copy(),self.high_vf.copy()

        # Low level optimizers
        self.low_policy_optimizer,self.low_Q1_optimizer,self.low_Q2_optimizer,self.low_vf_optimizer=\
            self.low_optimizers(optimizer_class,l_policy_lr,l_qf_lr,l_vf_lr)

        self.high_policy_optimizer, self.high_Q1_optimizer, self.high_Q2_optimizer, self.high_vf_optimizer = \
            self.high_optimizers(optimizer_class, h_policy_lr, h_qf_lr, h_vf_lr)

        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )


        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = False

    def _do_training(self, indices):

        mb_size=self.embedding_mini_batch_size
        num_updates= self.embedding_mini_batch_size

        # sample context batch
        context_batch= self.sample_context(indices)
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            self._update_low_level()
            context = context_batch[:, i *mb_size:i*mb_size + mb_size,:]
            self._take_step(indices,context)

            #stop backprop
            self.agent.detach_z()

    def _update_target_network(self,high=True):
        if high:
            ptu.soft_update_from_to(self.high_vf, self.high_vf_target, self.soft_target_tau)
        else:
            ptu.soft_update_from_to(self.low_vf,self.low_vf_target,self.soft_target_tau)

    def _update_low_level(self):

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_low_level()

        # run inference in networks
        policy_outputs = self.agent.forward_low(obs)#fix forward function to only take obs
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # flattens out the task dimension
        b, _ = obs.size()
        obs = obs.view(b, -1)
        actions = actions.view(b, -1)
        next_obs = next_obs.view(b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.low_Q1(obs, actions)
        q2_pred = self.low_Q2(obs, actions)
        v_pred = self.low_vf(obs)
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.low_vf_target(next_obs)


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.low_Q1_optimizer.zero_grad()
        self.low_Q2_optimizer.zero_grad()
        rewards_flat = rewards.view(b, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(b, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.high_Q1_optimizer.step()
        self.high_Q2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        q_pred_1=self.low_Q1(obs,new_actions)
        q_pred_2 = self.low_Q2(obs, new_actions)
        min_q_new_actions=torch.min(q_pred_1,q_pred_2)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.low_vf_optimizer.zero_grad()
        vf_loss.backward()
        self.low_vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.low_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.low_policy_optimizer.step()

    def _take_step(self,indices,context):
        #TODO: Need to change data collection to get full trajectory info
        #That way we can do the calculations needed for the trajectory optimization
        num_tasks = len(indices)
        # data is (task, batch, feat)

        obs, actions, rewards, next_obs, terms = self.sample_high_level(indices)
        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

    def augment_paths(self,paths):
        pass

    def sample_high_level(self,indices):
        '''Sample batch of high level interactions
        In the form of (Original state,Goal given,Reward received,State achieved)'''
        batches = [ptu.np_to_pytorch_batch(self.high_buffer.random_batch(idx, batch_size=self.high_batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_low_level(self):
        '''Sample batch of low level interactions
        In the form of ([state,goal], primitive action,parameterized reward, next state)'''
        batch = ptu.np_to_pytorch_batch(self.low_buffer.random_batch(batch_size=self.low_batch_size))
        unpacked = [self.unpack_batch(batch)] #puts it into format [o,a,r,s,d]
        # group like elements together
        unpacked = [[x[i][0] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self,indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        #This is a dope trick ^
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
                   idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    #TODO:
    # -Implement BURN version of sampling functions and batch unpacking
    # -Verify that you can get all the data necessary for _do_training_ and _do_eval_
    # -Implement training and eval functions (HARD)
    # -Fine tune
    # -Paydirt
    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''Gathers data and puts it in the buffers'''
        self.agent.clear_z()
        self.agent.clear_goal()

        num_transitions=0
        while num_transitions < num_samples:
            paths,n_samples=self.sampler.obtain_samples(max_samples=num_samples-num_transitions,
                                                       max_trajs=update_posterior_rate,
                                                       accum_context=False,
                                                       resample=resample_z_rate)
            num_transitions += n_samples
            low_level_paths=[{"observations":path["observations"],
                               "actions":path["actions"],
                               "rewards":path["param_rewards"],
                               "next_observations":path["next_observations"],
                               "terminals":path["low_level_terminals"],
                               "agent_infos":path["agent_infos"],
                               "env_infos":path["env_infos"]
                              } for path in paths]
            high_level_paths=[{"observations":path["meta_observations"],
                               "actions":path["goals"],
                               "rewards":path["meta_rewards"],
                               "next_observations":path["next_meta_observations"],
                               "terminals":path["meta_terminals"],
                               "agent_infos":[x for i,x in enumerate(path["agent_infos"]) if (i-1)%self.agent.c],
                               "env_infos":[x for i,x in enumerate(path["env_infos"]) if (i-1)%self.agent.c]
                              } for path in paths]

            enc_paths=[{"observations":path["pure_obs"],
                               "actions":path["actions"],
                               "rewards":path["rewards"],
                               "next_observations":path["next_pure_obs"],
                               "terminals":path["terminals"],
                               "agent_infos":path["agent_infos"],
                               "env_infos":path["env_infos"]}
                       for path in paths]
            for path in low_level_paths: #Single task buffer doesn't have an add paths method
                self.low_buffer.add_path(path)
            self.high_buffer.add_paths(self.task_idx,high_level_paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, enc_paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    def _can_train(self):
        return all([self.high_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks]+\
                   [self.low_buffer.num_steps_can_sample()>=self.low_batch_size])

    @property
    def networks(self):
        #Jesus...
        return self.agent.networks + [self.agent] + [self.high_Q1,self.high_Q2,self.low_Q1,self.low_Q2,self.high_vf,self.low_vf,self.high_vf_target,self.low_vf_target]

    def low_optimizers(self,optimizer_class,l_policy_lr,l_qf_lr,l_vf_lr):
        return  [optimizer_class(
                self.agent.low_level_policy.parameters(),
                lr=l_policy_lr,
            ),optimizer_class(
                self.low_Q1.parameters(),
                lr=l_qf_lr,
            ),
            optimizer_class(
                self.low_Q2.parameters(),
                lr=l_qf_lr,
            ), optimizer_class(
                self.low_vf.parameters(),
                lr=l_vf_lr,
            )]

    def high_optimizers(self, optimizer_class, h_policy_lr, h_qf_lr, h_vf_lr):
        return [optimizer_class(
            self.agent.high_level_policy.parameters(),
            lr=h_policy_lr,
        ), optimizer_class(
            self.high_Q1.parameters(),
            lr=h_qf_lr,
        ),
            optimizer_class(
                self.high_Q2.parameters(),
                lr=h_qf_lr,
            ), optimizer_class(
                self.high_vf.parameters(),
                lr=h_vf_lr,
            )]
