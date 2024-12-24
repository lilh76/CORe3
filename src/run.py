import os
import pprint
import time
import threading
import random
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.debug import *
from copy import deepcopy

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.transition_buffer import TransitionReplayBuffer
from learners.model_learner import ModelLearner

import json
import numpy as np



def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    logger.setup_sacred(_run)

    run_sequential(args=args, logger=logger)

    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    os._exit(os.EX_OK)


def evaluate(args, runner, mac):
    pass

def run_sequential(args, logger):
    FloatTensor = th.cuda.FloatTensor if args.use_cuda else th.FloatTensor

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    args.max_action = env_info.get("max_action", None)
    args.action_dim = env_info.get("action_dim", 1)
    args.action_space = runner.env._env.action_space

    mac = mac_REGISTRY[args.mac](args)
    learner = le_REGISTRY[args.learner](mac, None, logger, args)
    learner_model = ModelLearner(args, logger)
    if args.use_cuda:
        learner.cuda()
        learner_model.cuda()

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    start_time = time.time()
    last_time = start_time

    n_phase = len(args.learning_obj_lst)
    args.seen_obj = []

    start_phase = -1
    if args.checkpoint_path != '':
        t_env = eval(args.checkpoint_path.split('/')[-1])
        start_phase = 0
        t_sum = 0
        for t in args.t_max:
            t_sum += t
            if t_sum < t_env:
                start_phase += 1
            else:
                break

    for phase_idx in range(n_phase):
        args.learning_obj = args.learning_obj_lst[phase_idx] 
        args.n_learning_obj = len(args.learning_obj)

        for each_learning_obj in args.learning_obj:
            if each_learning_obj not in args.seen_obj:
                args.seen_obj.append(each_learning_obj) 
                
        args.learning_obj = sorted(args.learning_obj_lst[phase_idx], key=lambda x: args.seen_obj.index(x))
        args.n_seen_obj   = len(args.seen_obj)

        buffer = TransitionReplayBuffer(
            args,
            args.buffer_size,
            env_info["state_shape"],
            args.action_dim,
            args.n_learning_obj,
            args.n_seen_obj
        )

        runner.setup(mac=mac)

        if args.evaluate:
            evaluate(args, runner, mac)
            return

        for learning_obj in args.learning_obj_lst[phase_idx]:
            obj2head = mac.agent.obj2head if hasattr(mac.agent, "obj2head") else learner.critic1.obj2head
            if learning_obj not in obj2head:
                learner.create_head(learning_obj)
                learner_model.create_head(learning_obj, phase_idx)
        learner.set_optimiser()
        learner_model.set_optimiser(phase_idx)

        if phase_idx < start_phase:
            if phase_idx == start_phase - 1:
                mac.load_models(args.checkpoint_path)
                learner.load_models(args.checkpoint_path)
                learner_model.load_models(args.checkpoint_path)
            continue
                
        hypervolume_lst, sparsity_lst = runner.eval_agent_discrete()
        l = len(hypervolume_lst)
        for i in range(l):
            hv_log_name = f"hypervolume_task{i+1}" if i < l-1 else "hypervolume_all"
            sp_log_name = f"sparsity_task{i+1}" if i < l-1 else "sparsity_all"
            logger.log_stat(hv_log_name, hypervolume_lst[i], runner.t_env)
            logger.log_stat(sp_log_name, sparsity_lst[i], runner.t_env)

        t_env_start = runner.t_env
        last_learn_world_model_T = -args.learn_world_model_interval - 1

        while runner.t_env - t_env_start <= args.t_max[phase_idx]:
            
            if args.use_world_model: 
                preference = th.randn(args.n_seen_obj)
                preference = (th.abs(preference) / th.norm(preference, p=1)).type(FloatTensor)
            else:
                preference = th.zeros(args.n_seen_obj) 
                preference_learning_obj = th.randn(args.n_learning_obj)
                preference_learning_obj = (th.abs(preference_learning_obj) / th.norm(preference_learning_obj, p=1)) 
                index = np.where(np.isin(np.array(args.seen_obj), np.array(args.learning_obj)))[0]
                preference[index] = preference_learning_obj 
                preference = preference.type(FloatTensor)

            episode_batch = runner.run(preference=preference, eps_greedy_t=runner.t_env-t_env_start)
            buffer.push(episode_batch)
                
            if args.use_world_model and not args.oracle and \
                runner.t_env - last_learn_world_model_T >= args.learn_world_model_interval and \
                buffer.can_sample(args.batch_size_learn_world_model):
                for epoch in range(args.epoch_learn_world_model):
                    batch = buffer.sample(args.batch_size_learn_world_model)
                    learner_model.train_world_model(batch, runner.t_env, epoch)
                last_learn_world_model_T = runner.t_env
            if buffer.current_size >= args.batch_size * 6:
                for epoch in range(len(episode_batch['states'])):
                    batch = buffer.sample(args.batch_size)
                    learner.train(batch, runner.t_env, episode, learner_model)

            phase_end = sum([args.t_max[i] for i in range(phase_idx+1)])
            test_interval = args.test_interval if runner.t_env < phase_end - 5000 else 1000
            if (runner.t_env - last_test_T) / test_interval >= 1.0:
                logger.console_logger.info("Phase {} t_env: {} / {}".format(phase_idx, runner.t_env - t_env_start, args.t_max[phase_idx]))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(time_left(last_time,last_test_T,runner.t_env - t_env_start,args.t_max[phase_idx],),time_str(time.time() - start_time),))
                last_time = time.time()
                last_test_T = runner.t_env
                hypervolume_lst, sparsity_lst = runner.eval_agent_discrete()
                l = len(hypervolume_lst)
                for i in range(l):
                    hv_log_name = f"hypervolume_task{i+1}" if i < l-1 else "hypervolume_all"
                    sp_log_name = f"sparsity_task{i+1}" if i < l-1 else "sparsity_all"
                    logger.log_stat(hv_log_name, hypervolume_lst[i], runner.t_env)
                    logger.log_stat(sp_log_name, sparsity_lst[i], runner.t_env)
            

            episode += args.batch_size_run
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
                learner_model.save_models(save_path)

        hypervolume_lst, sparsity_lst = runner.eval_agent_discrete(final_test=True)
        l = len(hypervolume_lst)
        for i in range(l):
            hv_log_name = f"hypervolume_task{i+1}" if i < l-1 else "hypervolume_all"
            sp_log_name = f"sparsity_task{i+1}" if i < l-1 else "sparsity_all"
            logger.log_stat(hv_log_name, hypervolume_lst[i], runner.t_env)
            logger.log_stat(sp_log_name, sparsity_lst[i], runner.t_env)

        save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)
        learner_model.save_models(save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
