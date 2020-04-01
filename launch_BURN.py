"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.burn_sac import BURNSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.BURNAgent import BURNAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from configs.short_burn import short_config

def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )

    #low Qs first and then high Qs
    q_list=[
        [FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=2*obs_dim + action_dim,
        output_size=1,
    ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=2*obs_dim + action_dim,
            output_size=1,
        )]
        ,[FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    ),
    FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )]]
    #low vf first and then high vf
    vf_list = [FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=2*obs_dim,
        output_size=1,
    ),FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )]

    #NOTE: Reduced number of hidden layers in h_policy from 3 to 2 (idea being it's not doing as much as the whole policy in PEARL)
    h_policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=obs_dim,
    )
    #NOTE: Kept the 3 layers because fuck it it'll get tons of data
    l_policy=TanhGaussianPolicy(
        hidden_sizes=[net_size,net_size,net_size,net_size],
        obs_dim=2*obs_dim,
        latent_dim=0,
        action_dim=action_dim,
    )
    #TODO Implement BernAgent
    agent = BURNAgent(
        latent_dim,
        context_encoder,
        h_policy,
        l_policy,
        c=2,
        **variant['algo_params']
    )
    algorithm = BURNSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, q_list, vf_list],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    #TODO Make sure weights are properly saved
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        q_list[0][0].load_state_dict(torch.load(os.path.join(path, 'l_qf1.pth')))
        q_list[0][1].load_state_dict(torch.load(os.path.join(path, 'l_qf2.pth')))
        q_list[1][0].load_state_dict(torch.load(os.path.join(path, 'h_qf1.pth')))
        q_list[1][1].load_state_dict(torch.load(os.path.join(path, 'h_qf2.pth')))
        vf_list[0].load_state_dict(torch.load(os.path.join(path, 'l_vf.pth')))
        vf_list[1].load_state_dict(torch.load(os.path.join(path, 'h_vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        h_policy.load_state_dict(torch.load(os.path.join(path, 'h_policy.pth')))
        l_policy.load_state_dict(torch.load(os.path.join(path, 'l_policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = short_config
    print(variant['algo_params'])
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()
