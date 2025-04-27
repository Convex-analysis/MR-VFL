"""
Training script for the Mamba scheduler in MR-VFL.

This script implements a reinforcement learning approach to train the Mamba scheduler
for vehicle selection in federated learning. It uses the Proximal Policy Optimization (PPO)
algorithm to optimize the scheduler's performance.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
from tqdm import tqdm

# Import MR-VFL components
from mr_vfl_mamba_scheduler import MRVFLMambaActor, MRVFLMambaCritic, GlobalState
from environments.vehicular_fl_env import VehicularFLEnv
from environments.streaming_env import StreamingVehicularFLEnv
from environments.dynamic_env import DynamicVehicularFLEnv

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PPOMemory:
    """Memory buffer for PPO algorithm"""
    def __init__(self, batch_size=32):
        self.states = []
        self.vehicle_lists = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, vehicles, action, prob, val, reward, done):
        self.states.append(state)
        self.vehicle_lists.append(vehicles)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.vehicle_lists = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return batches

class MambaAgent:
    """Agent for training the Mamba scheduler using PPO"""
    def __init__(self, input_dim=6, state_dim=6, d_model=128, d_state=16,
                 actor_lr=0.0003, critic_lr=0.0003, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=32, n_epochs=10, entropy_coef=0.01):
        # Initialize actor and critic networks
        self.actor = MRVFLMambaActor(
            input_dim=input_dim,
            state_dim=state_dim,
            d_model=d_model,
            d_state=d_state
        ).to(device)

        self.critic = MRVFLMambaCritic(
            input_dim=input_dim,
            state_dim=state_dim,
            d_model=d_model,
            d_state=d_state
        ).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize memory
        self.memory = PPOMemory(batch_size)

        # Set hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef

        # Track training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []

    def store_transition(self, state, vehicles, action, prob, val, reward, done):
        """Store transition in memory"""
        self.memory.store(state, vehicles, action, prob, val, reward, done)

    def save_models(self, path):
        """Save actor and critic models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'model_state_dict': self.actor.state_dict(),  # For compatibility with MambaModel
        }, path)
        print(f"Models saved to {path}")

    def load_models(self, path):
        """Load actor and critic models"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Models loaded from {path}")
        else:
            print(f"No models found at {path}, using randomly initialized models")

    def select_action(self, vehicles, global_state, mask=None):
        """Select action using the actor network"""
        with torch.no_grad():
            # Get selection probabilities and continuous actions
            selection_probs, alpha, scheduled_time, bandwidth = self.actor(vehicles, global_state, mask)

            # Create categorical distribution
            dist = Categorical(selection_probs)

            # Sample action
            action = dist.sample()

            # Get log probability
            log_prob = dist.log_prob(action)

            # Get value
            action_tuple = (
                action.item(),
                alpha[action].item(),
                scheduled_time[action].item(),
                bandwidth[action].item()
            )
            value = self.critic(vehicles, global_state, action_tuple)

        return action.item(), log_prob.item(), value.item(), selection_probs.cpu().numpy()

    def learn(self):
        """Update actor and critic networks using PPO"""
        # Calculate advantages
        for _ in range(self.n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()

            # Process each batch
            for batch in batches:
                states = [self.memory.states[i] for i in batch]
                vehicle_lists = [self.memory.vehicle_lists[i] for i in batch]
                actions = [self.memory.actions[i] for i in batch]
                old_probs = [self.memory.probs[i] for i in batch]
                vals = [self.memory.vals[i] for i in batch]

                # Calculate advantages
                advantages = []
                returns = []
                for t in range(len(batch)):
                    idx = batch[t]
                    advantage = 0
                    discount = 1

                    # Calculate GAE
                    for k in range(idx, len(self.memory.rewards)):
                        if k == len(self.memory.rewards) - 1 or self.memory.dones[k]:
                            delta = self.memory.rewards[k] - vals[t]
                        else:
                            delta = self.memory.rewards[k] + self.gamma * self.memory.vals[k+1] - vals[t]

                        advantage += discount * delta
                        discount *= self.gamma * self.gae_lambda

                        if k == len(self.memory.rewards) - 1 or self.memory.dones[k]:
                            break

                    returns.append(advantage + vals[t])
                    advantages.append(advantage)

                advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
                returns = torch.tensor(returns, dtype=torch.float32).to(device)

                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Update actor and critic
                actor_loss_batch = []
                critic_loss_batch = []
                entropy_loss_batch = []
                total_loss_batch = []

                for i in range(len(batch)):
                    # Get current state, vehicles, and action
                    state = states[i]
                    vehicles = vehicle_lists[i]
                    action = actions[i]
                    old_prob = old_probs[i]

                    # Forward pass through actor
                    selection_probs, alpha, scheduled_time, bandwidth = self.actor(vehicles, state)

                    # Create categorical distribution
                    dist = Categorical(selection_probs)

                    # Get entropy
                    entropy = dist.entropy().mean()

                    # Get new log probability
                    new_prob = dist.log_prob(torch.tensor(action).to(device))

                    # Calculate probability ratio
                    prob_ratio = torch.exp(new_prob - torch.tensor(old_prob, dtype=torch.float32).to(device))

                    # Calculate weighted advantage
                    weighted_advantage = advantages[i] * prob_ratio

                    # Calculate clipped advantage
                    clipped_advantage = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[i]

                    # Calculate actor loss
                    actor_loss = -torch.min(weighted_advantage, clipped_advantage)

                    # Forward pass through critic
                    action_tuple = (
                        action,
                        alpha[action].item(),
                        scheduled_time[action].item(),
                        bandwidth[action].item()
                    )
                    value = self.critic(vehicles, state, action_tuple)

                    # Calculate critic loss
                    critic_loss = nn.MSELoss()(value, returns[i].unsqueeze(0))

                    # Calculate total loss
                    total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                    # Store losses
                    actor_loss_batch.append(actor_loss.item())
                    critic_loss_batch.append(critic_loss.item())
                    entropy_loss_batch.append(entropy.item())
                    total_loss_batch.append(total_loss.item())

                    # Backward pass
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                # Store average losses
                self.actor_losses.append(np.mean(actor_loss_batch))
                self.critic_losses.append(np.mean(critic_loss_batch))
                self.entropy_losses.append(np.mean(entropy_loss_batch))
                self.total_losses.append(np.mean(total_loss_batch))

        # Clear memory
        self.memory.clear()

def train_mamba_scheduler(args):
    """Train the Mamba scheduler using PPO"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("models", "trained", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize agent
    agent = MambaAgent(
        input_dim=args.input_dim,
        state_dim=args.state_dim,
        d_model=args.d_model,
        d_state=args.d_state,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef
    )

    # Load pretrained models if specified
    if args.load_model:
        agent.load_models(args.load_model)

    # Initialize environment based on scenario
    if args.scenario == "standard":
        env = VehicularFLEnv(
            vehicle_count=args.vehicle_count,
            max_round=args.max_round,
            sync_limit=args.sync_limit,
            traffic_density=args.traffic_density
        )
    elif args.scenario == "streaming":
        env = StreamingVehicularFLEnv(
            vehicle_count=args.vehicle_count,
            max_round=args.max_round,
            sync_limit=args.sync_limit,
            traffic_density=args.traffic_density
        )
    elif args.scenario == "dynamic":
        env = DynamicVehicularFLEnv(
            vehicle_count=args.vehicle_count,
            max_round=args.max_round,
            sync_limit=args.sync_limit,
            traffic_density=args.traffic_density
        )
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    # Initialize training metrics
    best_performance = 0
    episode_rewards = []
    episode_performances = []
    episode_lengths = []

    # Training loop
    for episode in range(args.n_episodes):
        # Reset environment
        env.reset()
        done = False
        episode_reward = 0
        step = 0

        # Create global state
        global_state = GlobalState()

        # Episode loop
        pbar = tqdm(total=args.max_round, desc=f"Episode {episode+1}/{args.n_episodes}")
        while not done and step < args.max_round:
            # Get eligible vehicles
            eligible_vehicles = [v for v in env.vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]

            if not eligible_vehicles:
                # No eligible vehicles, skip step
                reward = 0
                done = step >= args.max_round - 1
                step += 1
                pbar.update(1)
                continue

            # Convert vehicles to tensors
            vehicle_tensors = []
            for v in eligible_vehicles:
                vehicle_tensor = torch.tensor([
                    v.get('model_version', 0.5),
                    v['sojourn_time'],
                    v['computation_capacity'],
                    v['data_quality'],
                    v['channel_gain'],
                    0  # Default vehicle type
                ], dtype=torch.float32).to(device)
                vehicle_tensors.append(vehicle_tensor)

            # Convert global state to tensor
            global_state_tensor = global_state.to_tensor()

            # Select action
            action_idx, log_prob, value, _ = agent.select_action(vehicle_tensors, global_state_tensor)

            # Convert action index to vehicle index and get continuous actions
            vehicle_idx = eligible_vehicles[action_idx]['id']

            # Get continuous actions from the actor
            with torch.no_grad():
                _, alpha, scheduled_time, bandwidth = agent.actor(vehicle_tensors, global_state_tensor)
                alpha_value = alpha[action_idx].item()
                scheduled_time_value = scheduled_time[action_idx].item()
                bandwidth_value = bandwidth[action_idx].item()

            # Take step in environment with all required parameters
            _, reward, done, _ = env.step((vehicle_idx, alpha_value, scheduled_time_value, bandwidth_value))

            # Update global state
            global_state.scheduled_count += 1
            global_state.round_number = step
            global_state.current_model_performance = env.current_model_performance

            # Store transition
            agent.store_transition(global_state_tensor, vehicle_tensors, action_idx, log_prob, value, reward, done)

            # Update metrics
            episode_reward += reward
            step += 1
            pbar.update(1)

            # Learn if enough steps have been taken
            if step % args.update_interval == 0:
                agent.learn()

        pbar.close()

        # Learn at the end of the episode
        agent.learn()

        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_performances.append(env.current_model_performance)
        episode_lengths.append(step)

        # Print episode metrics
        print(f"Episode {episode+1}/{args.n_episodes} - Reward: {episode_reward:.2f}, Performance: {env.current_model_performance:.4f}, Length: {step}")

        # Save best model
        if env.current_model_performance > best_performance:
            best_performance = env.current_model_performance
            agent.save_models(os.path.join(output_dir, "best_model.pth"))
            print(f"New best performance: {best_performance:.4f}")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            agent.save_models(os.path.join(output_dir, f"checkpoint_{episode+1}.pth"))

        # Save final model
        if episode == args.n_episodes - 1:
            agent.save_models(os.path.join(output_dir, "final_model.pth"))

            # Also save to the standard location for the scheduler
            agent.save_models("models/mamba_scheduler.pth")

    # Print training summary
    print("\nTraining Summary:")
    print(f"Best Performance: {best_performance:.4f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Performance: {np.mean(episode_performances):.4f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")

    # Return trained agent
    return agent, {
        "best_performance": best_performance,
        "episode_rewards": episode_rewards,
        "episode_performances": episode_performances,
        "episode_lengths": episode_lengths,
        "actor_losses": agent.actor_losses,
        "critic_losses": agent.critic_losses,
        "entropy_losses": agent.entropy_losses,
        "total_losses": agent.total_losses
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Mamba scheduler for MR-VFL")

    # Environment parameters
    parser.add_argument("--scenario", type=str, default="standard", choices=["standard", "streaming", "dynamic"],
                        help="Training scenario")
    parser.add_argument("--vehicle_count", type=int, default=50, help="Number of vehicles in the environment")
    parser.add_argument("--max_round", type=int, default=100, help="Maximum number of rounds per episode")
    parser.add_argument("--sync_limit", type=int, default=1000, help="Synchronization time limit in seconds")
    parser.add_argument("--traffic_density", type=int, default=10, help="Traffic density (vehicles per km²)")

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=6, help="Input dimension for vehicle features")
    parser.add_argument("--state_dim", type=int, default=6, help="Input dimension for global state")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--d_state", type=int, default=16, help="State dimension for Mamba")

    # Training parameters
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Learning rate for actor")
    parser.add_argument("--critic_lr", type=float, default=0.0003, help="Learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--policy_clip", type=float, default=0.2, help="Policy clip parameter for PPO")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--update_interval", type=int, default=20, help="Number of steps between updates")
    parser.add_argument("--save_interval", type=int, default=10, help="Number of episodes between saving checkpoints")

    # Model loading/saving
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pretrained model")

    args = parser.parse_args()

    # Train the scheduler
    train_mamba_scheduler(args)

if __name__ == "__main__":
    main()
