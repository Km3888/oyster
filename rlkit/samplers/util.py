import numpy as np


def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = [] #low level observations (environment observation plus current goal)
    actions = [] #low level actions
    rewards = [] #low level rewards given by environment
    param_rewards=[] #low level reward given based on proximity to goal
    low_level_terminals = [] #denotes whether the transition ended a low-level rollout
    low_goals=[] #denotes the goal used by the low level policy at each timestep
    #note: because the goal here is actually the distance between the current state and
    #where we want to go, the low_goal during the rollout won't be the same as the original
    #goal given.
    real_terminals = []#denotes whether entire rollout ended
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0


    meta_obs=[] #State at the beginning of each high-level transition
    goals=[] #Goals outputted by high level policy
    meta_rewards=[] #Average reward accrued between high-level transitions
    meta_terminals=[] #
    pure_obs=[]

    curr_meta_obs=o
    next_meta_obs=None
    curr_meta_reward = 0
    low_steps_taken=0

    if agent.use_goals:
        curr_goal=agent.sample_goal(o)
        low_goal=curr_goal
    if animated:
        env.render()
    while path_length < max_path_length:
        #the code I added
        if agent.use_goals and (low_steps_taken==agent.c or np.linalg.norm(low_goal-o)<agent.epsilon):
            #Record data for high level policy and choose a new goal
            next_meta_obs=o

            meta_obs.append(curr_meta_obs)
            goals.append(curr_goal)
            meta_rewards.append(curr_meta_reward)
            meta_terminals.append(d)
            curr_meta_reward=0
            curr_meta_obs=next_meta_obs
            curr_goal=agent.sample_goal(curr_meta_obs)

            low_goal=curr_goal
            #TODO
            #So there's kind of a problem in that we have to change the internal goal of the agent during the
            #low level rollout but we want it to stay the same so that we can gather the transition
            low_steps_taken=0

        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        #goal transition
        if agent.use_goals:
            low_steps_taken += 1
            next_low_goal=next_o+low_goal-o
            agent.goal=next_low_goal
            curr_meta_reward+=r
        # update the agent's context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])

        rewards.append(r)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        real_terminals.append(d)

        path_length += 1
        if agent.use_goals:
            pure_obs.append(o)
            observations.append(np.concatenate([o, low_goal]))
            param_rewards.append(-np.linalg.norm((o+low_goal)-next_o))
            low_level_terminals.append(low_steps_taken==agent.c)
            low_goals.append(low_goal)
            low_goal = next_low_goal
        else:
            observations.append(o)
        if d:
            break
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image

    next_goal=agent.sample_goal(next_o)
    meta_obs+=[curr_meta_obs,next_o]
    if agent.use_goals:
        goals.append(agent.goal)
        meta_rewards.append(curr_meta_reward)
        meta_terminals.append(d)
        full_next_o=np.concatenate([next_o,agent.goal])
    else:
        full_next_o=next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    pure_obs=np.array(pure_obs)
    meta_observations=np.array(meta_obs)
    goals=np.array(goals)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        full_next_o = np.array(full_next_o)
    if len(meta_observations.shape)==1 and agent.use_goals:
        meta_observations=np.expand_dims(meta_observations,1)
        next_meta_observations=np.array([next_o]+[agent.goal])
    if len(pure_obs.shape) == 1 and agent.use_goals:
        observations = np.expand_dims(pure_obs, 1)
        full_next_o = np.concatenate([np.array(next_o),np.array(next_goal)])


    if agent.use_goals:
        next_meta_observations=np.vstack(
            (
                meta_observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        next_pure_obs=np.vstack(
            (
                pure_obs[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        next_observations = np.vstack(
            (
                observations[1:, :],
                full_next_o,
                np.expand_dims(full_next_o, 0)
            )
        )
    else:
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )
        next_pure_obs=[]
        next_meta_observations=[]
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        param_rewards=np.array(param_rewards).reshape(-1,1),
        next_observations=next_observations,
        low_level_terminals=np.array(low_level_terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        terminals=real_terminals,
        pure_obs=pure_obs,
        next_pure_obs=next_pure_obs,

        meta_observations=meta_observations,
        goals=goals,
        meta_rewards=np.array(meta_rewards).reshape(-1,1),
        next_meta_observations=next_meta_observations,
        meta_terminals=np.array(meta_terminals).reshape(-1,1)#Meta terminals are the same thing as real_terminals but at a lower frequency
        #I can't be bothered to do infos for the meta policy currently, may come back to it
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
