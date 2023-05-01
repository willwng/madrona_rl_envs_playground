# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.hanabi_env import HanabiMadrona, config_choice, PantheonHanabi
from pantheonrl_extension.vectorenv import SyncVectorEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="CartPole-v1",
    #     help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=6.25e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1280,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=80,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(torch.jit.ScriptModule):
    def __init__(self, envs, hid_dim, num_ff_layers):
        super().__init__()

        infeat = int(np.array(envs.observation_space.shape).prod())
        print(infeat)
        ff_layers = [nn.Linear(infeat, 512), nn.ReLU()]
        for i in range(num_ff_layers):
            ff_layers.append(nn.Linear(hid_dim, hid_dim))
            ff_layers.append(nn.ReLU())

        self.common = nn.Sequential(*ff_layers)
        self.critic = nn.Linear(hid_dim, 1)
        self.actor = nn.Linear(hid_dim, envs.action_space.n)

    @torch.jit.script_method
    def get_value(self, x: torch.Tensor):
        return self.critic(self.common(x))

    @torch.jit.script_method
    def get_action_and_value(self, x: torch.Tensor, action_mask: torch.Tensor):
        com = self.common(x)
        logits = self.actor(com)
        logits[torch.logical_not(action_mask)] = -1e10

        prob_a = nn.functional.softmax(logits, 1)
        log_pa = nn.functional.log_softmax(logits, 1)
        action = prob_a.multinomial(1)

        ent = (-log_pa * prob_a).sum(1)
        log_pa = log_pa.gather(1, action).squeeze(1)
        action = action.squeeze(1)

        return action, log_pa, ent, self.critic(com)

    @torch.jit.script_method
    def get_value_from_action(self, x: torch.Tensor, action_mask: torch.Tensor, action: torch.Tensor):
        com = self.common(x)
        logits = self.actor(com)
        logits[torch.logical_not(action_mask)] = -1e10

        prob_a = nn.functional.softmax(logits, 1)
        log_pa = nn.functional.log_softmax(logits, 1)
        action = prob_a.multinomial(1)

        ent = (-log_pa * prob_a).sum(1)
        log_pa = log_pa.gather(1, action).squeeze(1)
        action = action.squeeze(1)

        return action, log_pa, ent, self.critic(com)

def make_env(seed, idx):
    def thunk():
        env = PantheonHanabi({
            "colors":
                5,
            "ranks":
                5,
            "players":
                2,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type": 1
        })
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"Hanabi__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = HanabiMadrona(args.num_envs, 0, False, config={
            "colors":
                5,
            "ranks":
                5,
            "players":
                2,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type": 1
        })

    # envs = SyncVectorEnv(
    #         [make_env(args.seed + i, i) for i in range(args.num_envs)]
    #     )
    n_players = envs.n_players
    agent = Agent(envs, 512, 2).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1.5e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((n_players, args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    if envs.action_space.shape == tuple():
        actions = torch.zeros((n_players, args.num_steps, args.num_envs, 1)).to(device, dtype=torch.long)
    else:
        actions = torch.zeros((n_players, args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    action_masks = torch.zeros((n_players, args.num_steps, args.num_envs, envs.action_space.n)).to(device, dtype=torch.bool)
    logprobs = torch.zeros((n_players, args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((n_players, args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((n_players, args.num_steps, args.num_envs)).to(device)

    active = torch.zeros((n_players, args.num_steps, args.num_envs), dtype=torch.bool).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.n_reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    running_scores = torch.zeros(args.num_envs)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        score_sum = 0
        num_scores = 0

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            for p in range(n_players):
                obs[0, step][next_obs[p].active.to(torch.bool)] = next_obs[p].obs[next_obs[p].active.to(torch.bool)].to(torch.float)
                action_masks[0, step][next_obs[p].active.to(torch.bool)] = next_obs[p].action_mask[next_obs[p].active.to(torch.bool)]
                active[0, step][next_obs[p].active.to(torch.bool)] = next_obs[p].active[next_obs[p].active.to(torch.bool)]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # action = []
                for p in range(n_players):
                    action_i, logprob, _, value = agent.get_action_and_value(next_obs[p].obs.float(), next_obs[p].action_mask)
                    values[0, step][next_obs[p].active.to(torch.bool)] = value.flatten()[next_obs[p].active.to(torch.bool)]
                    actions[0, step][next_obs[p].active.to(torch.bool)] = action_i[next_obs[p].active.to(torch.bool)].unsqueeze(-1)
                    logprobs[0, step][next_obs[p].active.to(torch.bool)] = logprob[next_obs[p].active.to(torch.bool)]
                actions[1, step] = actions[0, step]
                # action.append(action_i)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, _ = envs.n_step(actions[:, step])
            rewards[:, step] = reward  # + carry_rewards
            # carry_rewards[active[:, step]] = 0
            # carry_rewards[~active[:, step]] = rewards[:, step][~active[:, step]]
            running_scores[:] += rewards[0, step] # only track ego

            # if torch.any(running_scores < 0):
            #     print((running_scores < 0).nonzero(as_tuple=True), step, running_scores[running_scores < 0])

            if torch.any(next_done == 1):
                score_sum += running_scores[next_done == 1].sum()
                num_scores += next_done.count_nonzero()
                running_scores[next_done == 1] = 0

        # print(rewards[0, :, 0])
        # print(dones[:, 0])
        # temprew = 0
        # for i in range(args.num_steps):
        #     if dones[i, 0]:
        #         print(temprew)
        #         temprew = 0
        #     temprew += rewards[0, i, 0]

        # bootstrap value if not done
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            for p in range(1):
                next_value = agent.get_value(next_obs[p].obs.float()).reshape(-1)
                # lastgaelam = 0
                delta = torch.zeros(args.num_envs).to(device)

                bootstrapped = next_obs[p].active.detach().clone().to(dtype=torch.bool)
                nextnonterminal = torch.zeros(args.num_envs).to(device)
                nextvalues = torch.zeros(args.num_envs).to(device)

                running_rewards = torch.zeros(args.num_envs).to(device)

                lastgaelam = torch.zeros(args.num_envs).to(device)
                nextnonterminal = 1.0 - next_done.to(torch.float)
                nextvalues = next_value
                for t in reversed(range(args.num_steps)):
                    mask = active[p, t]
                    computemask = mask.to(torch.bool)

                    # if not torch.all(bootstrapped):
                    #     computemask = mask & bootstrapped.logical_not()

                    #     bootstrapped |= mask

                    #     # disable advantages for these final bootstrapped values
                    #     active[p, t, computemask] = False
                    running_rewards += rewards[p, t]
                    delta[computemask] = running_rewards[computemask] + args.gamma * nextvalues[computemask] * nextnonterminal[computemask] - values[p, t, computemask]
                    advantages[p, t, computemask] = lastgaelam[computemask] = delta[computemask] + args.gamma * args.gae_lambda * nextnonterminal[computemask] * lastgaelam[computemask]

                    running_rewards[dones[t].to(torch.bool) | mask] = 0
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues[mask] = values[p, t, mask]
            returns = advantages + values

        # print(active.sum(2))
        # print(returns)

        # print(values.flatten().shape)

        # flatten the batch
        b_obs = obs[active].reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs[active].reshape(-1)
        b_actions = actions[active].reshape((-1,) + envs.action_space.shape)
        b_action_masks = action_masks[active].reshape((-1, envs.action_space.n))
        b_advantages = advantages[active].reshape(-1)
        b_returns = returns[active].reshape(-1)
        b_values = values[active].reshape(-1)

        # print("VERSUS", b_values.shape)
        # print(b_returns)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # mb_inds = torch.randperm(b_values.size(0))

            _, newlogprob, entropy, newvalue = agent.get_value_from_action(b_obs, b_action_masks, b_actions.long())
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            # if args.clip_vloss:
            #     v_loss_unclipped = (newvalue - b_returns) ** 2
            #     v_clipped = b_values + torch.clamp(
            #         newvalue - b_values,
            #         -args.clip_coef,
            #         args.clip_coef,
            #     )
            #     v_loss_clipped = (v_clipped - b_returns) ** 2
            #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            #     v_loss = 0.5 * v_loss_max.mean()
            # else:
            #     v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            v_loss = nn.functional.smooth_l1_loss(newvalue, b_returns, reduction="mean")

            entropy_loss = entropy.mean()
            loss = (pg_loss - args.ent_coef * entropy_loss + v_loss) * 128

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # print("losses/value_loss", v_loss.item(), global_step)
        # print("losses/policy_loss", pg_loss.item(), global_step)
        # print("losses/entropy", entropy_loss.item(), global_step)
        # print("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # print("losses/approx_kl", approx_kl.item(), global_step)
        # print("losses/clipfrac", np.mean(clipfracs), global_step)
        # print("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("Score:", score_sum/num_scores if num_scores != 0 else "NAN", num_scores, rewards[1].mean())

    envs.close()
    writer.close()
