from agents.ddpg.modules.buffer import *
from agents.ddpg.modules.utils import *
from agents.ddpg.modules.backbone import *
from agents.ddpg.modules.core import *
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class DDPGAgent:
    """DDPGAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            args,
            env
    ):
        """Initialize."""
        self.obs_dim = env.observation_space.shape[0]
        print(env.observation_space.shape)
        self.action_dim = env.action_space.shape[0]
        print(env.action_space.shape)

        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.ou_noise_theta = args.ou_theta
        self.ou_noise_sigma = args.ou_sigma
        self.gamma = args.gamma
        self.tau = args.tau
        self.env = env
        self.memory = ReplayBuffer(self.obs_dim, self.action_dim, self.memory_size, self.batch_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.initial_random_steps = args.initial_steps

        # noise
        self.noise = OUNoise(
            self.action_dim,
            theta=self.ou_noise_theta,
            sigma=self.ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.obs_dim + self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim + self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0
        self.episode = 0
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()

        # add noise for exploration during training
        # if not self.is_test:
        #     noise = self.noise.sample()
        #     selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        # self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        state_next, reward, done, info = self.env.step(action)

        # if not self.is_test:
        #     self.transition += [reward, state_next, done, info]
        #     print(self.transition)
        #     self.memory.store(*self.transition)

        return state_next, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        state_next = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        info = torch.FloatTensor(samples["false"].reshape(-1, 1)).to(device)
        if 0 == True:
            print(f"state size: {np.shape(state)}")

        masks = 1 - done
        next_action = self.actor_target(state_next)
        # print(f"next action: {np.shape(next_action)}")
        next_value = self.critic_target(state_next, next_action)
        curr_return = reward + self.gamma * next_value * masks

        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def train(self, args):
        num_ep = args.max_episode
        num_frames = args.max_step
        plotting_interval = args.plot_interval

        """Train the agent."""
        for self.episode in range(1, num_ep + 1):
            self.is_test = False

            state = self.env.reset()

            actor_losses = []
            critic_losses = []
            scores = []
            score = 0
            for self.total_step in range(1, num_frames + 1):
                action = self.select_action(state)
                state_next, reward, done, info = self.step(action)
                # state_next = state_next.squeeze()
                print(f"reward of step {self.total_step} in episode{self.episode} is: {reward}")
                state = state_next

                score = score + reward
                # if episode ends
                if done:
                    state = self.env.reset()
                    scores.append(score)
                    score = 0

                # if training is ready
                if (
                        len(self.memory) >= self.batch_size
                        and self.total_step > self.initial_random_steps
                ):
                    actor_loss, critic_loss = self.update_model()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                # plotting
                if self.total_step % plotting_interval == 0:
                    self._plot(
                        self.total_step,
                        scores,
                        actor_losses,
                        critic_losses,
                    )
                    pass
        self.env.close()

    def test(self):
        # """Test the agent."""
        # self.is_test = True

        # state = self.env.reset()
        # done = False
        # score = 0

        # frames = []
        # while not done:
        #     frames.append(self.env.render(mode="rgb_array"))
        #     action = self.select_action(state)
        #     next_state, reward, done = self.step(action)

        #     state = next_state
        #     score = self.discount*score + reward

        # print("score: ", score)
        # self.env.close()

        # return frames
        pass

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()