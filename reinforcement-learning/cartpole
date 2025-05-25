import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
import random
from matplotlib import animation
from IPython.display import display, clear_output
#----------------------------------------------------------------#

#----------------------------------------------------------------#
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.reset()

num_steps_to_viz = 200
step_count = 0
for i in range(num_steps_to_viz):
   step_count += 1
   if(i % 3 == 0): # Speed up the framerate
      plt.imshow(env.render())
      display(plt.gcf())
      clear_output(wait=True)

   _, _, terminated, truncated, _ =  env.step(random.randrange(0,2)) # take a random action
   done = terminated or truncated
   if done:
      print("Pole tipped too far")
      print(f"Survived for {step_count} steps")
      break

env.close()
#----------------------------------------------------------------#

class DQN(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(DQN, self).__init__()

    self.l1 = nn.Linear(state_dim, 128)
    self.l2 = nn.Linear(128, 128)
    self.l3 = nn.Linear(128, action_dim)

    self.relu = nn.ReLU();

  def forward(self, input):
    x = self.l1(input)
    x = self.relu(x)
    x = self.l2(x)
    x = self.relu(x)
    x = self.l3(x);
    return x;

#----------------------------------------------------------------#
from collections import namedtuple, deque

# Creates a named tuple that we can add to
Transition = namedtuple(
    "Transition",
    ('state_a', 'action', 'state_b', 'reward')
)

# Example of creating a named tuple

t = Transition([0,0,0,0], 1, [1,1], 0.5)

# You can check the value of, say, "action" by printing t.action

#----------------------------------------------------------------#

class TransitionMemoryStorage():
    """A class to hold a buffer of transition tuples that can be sampled from to run experience replay on a DQN agent
    """
    def __init__(self, capacity):
        """Creates a buffer to hold transition tuples

        Args:
            capacity (int): How many elements the buffer can hold at a time
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add_transition(self, t):
        self.buffer.append(t)

        """Adds a transition to the buffer

        Args:
            t (tuple): A Transition tuple
        """


    def sample(self, num_samples):
        return random.sample(list(self.buffer), num_samples)

        """Selects num_samples unique samples from the buffer

        Args:
            num_samples (int): Number of samples to pull

        Returns:
            list: Sample list of transitions from the buffer
        """


    def can_sample(self, num_samples):
        return len(self.buffer) >= num_samples

        """Checks if there are at least num_samples samples in the buffer
        Args:
            num_samples (int): How many samples to check validity for

        Returns:
            boolean: If the TransitionMemoryStorage object can be sampled on
        """

import math
class EpsilonGreedyStrategy():
    """Strategy for use in an agent, implements exponential decay epsilon greedy
    """
    def __init__(self, max_epsilon, min_epsilon, decay):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        """Initializes an Epsilon Greedy strategy

        Args:
            max_epsilon (float): The initial epsilon value
            min_epsilon (float): The ending epsilon value
            decay (float): The rate at which the epsilon value will decay
        """


    def should_explore(self, current_step):
        return random.random() < self._get_explore_prob(current_step)
        """Returns True if, according to this strategy at the current timestep, the agent should explore and False otherwise

        Args:
            current_step (int): How many steps the agent has taken (persists through episodes and failures)

        Returns:
            boolean: If the agent should explore (take a random action)
        """


    def _get_explore_prob(self, current_step):
        return ((self.max_epsilon - self.min_epsilon) * math.exp(-self.decay*current_step)) + self.min_epsilon
        """Returns the epsilon value at a certain timestep

        Args:
            current_step (int): The number of steps that the agent has taken

        Returns:
            float: Epsilon value at this current timestep
        """

class Agent():
    """Agent that acts within the environment using a dqn policy
    """
    def __init__(self, strategy):
        """Initializes the agent with a strategy for explore vs. exploit

        Args:
            strategy: Strategy object that dictates to explore or exploit
        """
        self.strategy = strategy
        self.current_step = 0

    def select_action(self, input_state, policy_dqn):
        self.current_step += 1
        if self.strategy.should_explore(self.current_step):
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                return torch.argmax(policy_dqn(input_state)).item()


        """Selects an action based on a state and a policy network

        Args:
            input_state: The state to select an action based on
            policy_dqn: Policy network that outputs probabilities to take actions within the action space

        Returns:
            int: Action to take based on state
        """

#----------------------------------------------------------------#
from itertools import count
# Instantiate an EpsilonGreedyStrategy class
e_greedy_strategy = EpsilonGreedyStrategy(max_epsilon=0.9, min_epsilon=0.05, decay=0.99)

# Instantiate an agent with the above explore/exploit strategy
agent = Agent(strategy=e_greedy_strategy)

# Create a TransitionMemoryStorage object
memory = TransitionMemoryStorage(1000)

# Create a policy and target net
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)

# Set up an optimizer and loss function
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)
# This is Huber loss, The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large.
loss_func = nn.SmoothL1Loss()

BATCH_SIZE = 128
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Target network soft update factor

def optimize_model():
    # If the amount of Transitions saved in memory is not big enough to get a batch from, don't sample
    if not memory.can_sample(BATCH_SIZE):
        return
    transitions = memory.sample(BATCH_SIZE)

    # Batching data for easier processing
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state_b)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.state_b if s is not None])
    state_batch = torch.cat(batch.state_a)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    ## Recall the Bellman Equation: Q(s, a) = r + (GAMMA * Q(s', a)) where a is the action selected by the agent
    ## The loss we are trying to compute (a.k.a TD-error) is: loss = Q(s, a) - [r + (GAMMA * Q(s', a))]

    # This is Q(s, a), the current Q-values
    predicted_q_values = policy_net(state_batch).gather(1, action_batch)

    # These lines calculate Q(s', a) where a is the action selected by the agent
    next_state_q_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad(): # This is needed because we don't want to pass in "None" values to our target_net as it would crash
        next_state_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # This is r + (GAMMA * Q(s', a))
    expected_q_values = reward_batch + (next_state_q_values * GAMMA)

    # Compute Huber loss
    loss = loss_func(predicted_q_values, expected_q_values.unsqueeze(1))

    # Optimize the model (standard procedure)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target(policy_net, target_net):
    """ Soft update of the target network's weights. target_net ← (policy_net*τ) + [target_net*(1-τ)]
    Args:
        policy_net : Policy DQN used to select actions
        target_net : Target DQN used to prevent self-optimization loop
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

    target_net.load_state_dict(target_net_state_dict)

num_episodes = 1000 # Number of episodes to run, can be between 200 to 1000
for i_episode in range(num_episodes):
    if(i_episode % 25 == 0): # Sanity check every 25 episodes
        print(f"On episode {i_episode}")

    # Get the initial state of the environment
    current_state, _ = env.reset()
    # Reshape into a tensor of shape (1, 4) which is necessary for batching
    current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

    done = False
    while not done:
        # Select an action
        action = agent.select_action(current_state, policy_net)
        # Take a step in the environment and get the results from that step
        next_state, reward, terminated, truncated, _ = env.step(action)

        # We do not want to consider terminated transition states. This is used to mask data for model updates
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)


        # Store the transition in memory, reshaping for batching
        memory.add_transition(Transition(current_state, torch.tensor([action]).unsqueeze(0), next_state, torch.tensor([reward])))

        # Move to the next state
        current_state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        update_target(policy_net, target_net)

        # Terminated means the agent "wins" or "loses", truncated means some other failure happened. Either way, we move on
        done = terminated or truncated

print('Complete')

#----------------------------------------------------------------#
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

num_trials = 10000
overall_max_reward = 0

done = False
for i in range(num_trials):
    episode_reward = 0
    for t in count():
        action = agent.select_action(state, policy_net)
        state,reward,terminated,truncated,_ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = terminated or truncated

        episode_reward+=1
        overall_max_reward = max(episode_reward, overall_max_reward)

        if done:
            break

print(f"Longest time alive across {num_trials} trials: {overall_max_reward}")
#----------------------------------------------------------------#
