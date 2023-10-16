import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class Agent(nn.Module):
    def __init__(self, n_actions):
        super(Agent, self).__init__()
        #self.register_buffer("highscore", T.tensor([0]))
        
        self.actor_cnn = nn.Sequential(
            ##  input shape: 1x96x96
            nn.Conv2d(1, 32, 5, 1), 
            ## input shape: 3x92x92
            nn.MaxPool2d(4,4),
            nn.LeakyReLU(inplace=True),
            ## input shape: 3x23x23
            nn.Conv2d(32, 64, 4, 1),
            ## input shape: 12x20x20
            nn.MaxPool2d(4,4),
            nn.LeakyReLU(inplace=True),
            ## input shape: 64x5x5
        )
        
        self.actor_fc = nn.Sequential(
            layer_init(nn.Linear(1600, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01)
        )
        
        self.critic_cnn = nn.Sequential(
            ##  input shape: 1x96x96
            nn.Conv2d(1, 32, 5, 1), 
            ## input shape: 3x92x92
            nn.MaxPool2d(4,4),
            nn.LeakyReLU(inplace=True),
            ## input shape: 3x23x23
            nn.Conv2d(32, 64, 4, 1),
            ## input shape: 12x20x20
            nn.MaxPool2d(4,4),
            nn.LeakyReLU(inplace=True),
            ## input shape: 64x5x5
        )
        
        self.critic_fc = nn.Sequential(
            layer_init(nn.Linear(1600, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1)
        )
        
    def get_action_and_value(self, x):
        a_state = self.actor_cnn(x)
        v_state = self.critic_cnn(x)
        a_state = a_state.view(-1, 1600)
        v_state = v_state.view(-1, 1600)
        logits = self.actor_fc(a_state)
        value = self.critic_fc(v_state)
        
        return logits, value
    
class PPOTrainer():
    def __init__(self, 
                 agent,
                 ppo_clip_val=0.1,
                 target_kl_div=0.02,
                 train_iters=4,
                 lr=1e-4,
                 entropy_coefficient=1e-3,
                 value_coefficient=0.5,
                 grad_max_norm=0.5,
                 chkpt_dir="./models/"):
        self.agent = agent
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.train_iters = train_iters
        self.checkpoint_file_best = chkpt_dir + "ppo_CNN_v6_best.pth"
        self.checkpoint_file_current = chkpt_dir + "ppo_CNN_v6.pth"
        self.checkpoint_file_old = chkpt_dir + "ppo_CNN_v5_best.pth"
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.grad_max_norm = grad_max_norm
        
        agent_params = list(self.agent.parameters())
        self.optim = optim.AdamW(agent_params, lr=lr, eps=1e-7)
        
        self.scaler = T.cuda.amp.GradScaler()
    
    def train(self, obs, acts, old_log_probs, gaes, returns, batches):
        try:
            kl_divs = []
            avg_total_loss = np.array([])
            avg_policy_loss = np.array([])
            avg_value_loss = np.array([])
            for i in range(self.train_iters):   
                batched_obs_list = T.split(obs, int(len(obs)/batches))
                batched_old_probs_list = T.split(old_log_probs, int(len(old_log_probs)/batches))
                batched_acts_list = T.split(acts, int(len(acts)/batches))
                batched_gaes_list = T.split(gaes, int(len(gaes)/batches))
                batched_returns_list = T.split(returns, int(len(returns)/batches))
                
                for i in range(batches):                    
                    ## following the surrogate loss formula
                    batched_obs = batched_obs_list[i]
                    batched_old_probs = batched_old_probs_list[i]
                    batched_acts = batched_acts_list[i]
                    batched_gaes = batched_gaes_list[i]
                    batched_returns = batched_returns_list[i]
                
                    with T.cuda.amp.autocast():
                        new_logits, values = self.agent.get_action_and_value(batched_obs)
                        new_logits = Categorical(logits=new_logits)
                        new_log_probs = new_logits.log_prob(batched_acts)
                        
                        ## getting the ratio of the new policy divided by the old policy
                        policy_ratio = T.exp(new_log_probs - batched_old_probs)
                        
                        ## clip the ratio in between 0.8 and 1.2
                        clipped_ratio = policy_ratio.clamp(
                            1-self.ppo_clip_val, 1+self.ppo_clip_val
                        )
                        
                        clipped_loss = -clipped_ratio*batched_gaes
                        normal_loss = -policy_ratio*batched_gaes
                    
                        ## get the smaller of the 2 values - ppo wants to 
                        ## make smaller rather than bigger steps
                        policy_loss = T.max(normal_loss, clipped_loss).mean()
                        ploss_log = T.max(normal_loss, clipped_loss).mean().item()
                        
                        ## L2 loss
                        value_loss = 0.5 * T.mean((batched_returns - values) ** 2)
                        vloss_log = 0.5 * T.mean((batched_returns - values) ** 2).item()
                        
                        total_loss = policy_loss - self.entropy_coefficient * T.mean(new_logits.entropy()) + value_loss * self.value_coefficient   
                        tloss_log = (policy_loss - self.entropy_coefficient * T.mean(new_logits.entropy()) + value_loss * self.value_coefficient).item()   
                        
                        avg_policy_loss = np.append(avg_policy_loss, ploss_log)
                        avg_value_loss = np.append(avg_value_loss, vloss_log)
                        avg_total_loss = np.append(avg_total_loss, tloss_log)  
                        
                    self.optim.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    nn.utils.clip_grad.clip_grad_norm_(self.agent.parameters(), max_norm=self.grad_max_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                 

                    ## add KL divergence check to prevent the policy from changing too much
                    ## monte-carlo estimate, not exact
                    kl_div = (batched_old_probs-new_log_probs).mean()
                    kl_divs.append(kl_div.cpu().detach().item())
                    
                    if kl_div >= self.target_kl_div:
                        print('KL Divergence exceeded')
                        break

            return True, kl_divs, avg_policy_loss.mean(), avg_value_loss.mean(), avg_total_loss.mean()
        
        ## if user does not have a gpu/it doesnt work for them     
        except Exception as e:
            print(e)
            try:
                kl_divs = []
                avg_total_loss = np.array([])
                avg_policy_loss = np.array([])
                avg_value_loss = np.array([])
                for _ in range(self.policy_train_iters):   
                    for i in range(batches):                    
                        ## following the surrogate loss formula
                        batched_obs = obs[int(len(obs)/batches)*(i-1):int(len(obs)/batches)*i-1]
                        batched_old_probs = old_log_probs[int(len(obs)/batches)*(i-1):int(len(obs)/batches)*i-1]
                        
                        new_logits, values = self.agent.get_action_and_value(batched_obs)
                        new_logits = Categorical(logits=new_logits)
                        new_log_probs = new_logits.log_prob(acts)
                        
                        ## getting the ratio of the new policy divided by the old policy
                        policy_ratio = T.exp(new_log_probs - old_log_probs[batched_old_probs])
                        
                        ## clip the ratio in between 0.8 and 1.2
                        clipped_ratio = policy_ratio.clamp(
                            1-self.ppo_clip_val, 1+self.ppo_clip_val
                        )
                        
                        clipped_loss = -clipped_ratio*gaes
                        full_loss = -policy_ratio*gaes
                        
                        ## get the smaller of the 2 values - ppo wants to 
                        ## make smaller rather than bigger steps
                        policy_loss = T.max(full_loss, clipped_loss).mean()
                        
                        ## L2 loss
                        value_loss = 0.5 * ((returns - values) ** 2).mean()
                        
                        total_loss = policy_loss + self.entropy_coefficient * new_logits.entropy() + value_loss * self.value_coefficient
                        
                        self.optim.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad.clip_grad_norm_(self.agent.parameters(), max_norm=self.grad_max_norm)
                        self.optim.step()
                                           
                    ## add KL divergence check to prevent the policy from
                    ## changing too much - which could ruin the model
                    ## monte-carlo estimate, not exact
                    kl_div = (old_log_probs-new_log_probs).mean()
                    kl_divs.append(kl_div)
                    
                    if kl_div >= self.target_kl_div:
                        print('KL Divergence exceeded')
                        break
                return True, kl_divs, avg_policy_loss.mean(), avg_value_loss.mean(), avg_total_loss.mean()
            except Exception:
                return False, kl_divs, avg_policy_loss.mean(), avg_value_loss.mean(), avg_total_loss.mean()
    
    def train_policy(self, obs, acts, old_log_probs, gaes, returns, batches):
        try:
            kl_divs = []
            for i in range(self.policy_train_iters):   
                for i in range(batches):
                    self.policy_optim.zero_grad()
                    
                    ## following the surrogate loss formula
                    batched_obs = obs[int(len(obs)/batches)*(i-1):int(len(obs)/batches)*i-1]
                    batched_old_probs = old_log_probs[int(len(obs)/batches)*(i-1):int(len(obs)/batches)*i-1]
                    
                    with T.cuda.amp.autocast():
                        new_logits = self.actor.layers(batched_obs)
                        new_logits = Categorical(logits=new_logits)
                        new_log_probs = new_logits.log_prob(acts)
                        
                        ## getting the ratio of the new policy divided by the old policy
                        policy_ratio = T.exp(new_log_probs - old_log_probs[batched_old_probs])
                        
                        ## clip the ratio in between 0.8 and 1.2
                        clipped_ratio = policy_ratio.clamp(
                            1-self.ppo_clip_val, 1+self.ppo_clip_val
                        )
                        
                        clipped_loss = clipped_ratio*gaes
                        full_loss = policy_ratio*gaes
                    
                        ## get the smaller of the 2 values - ppo wants to 
                        ## make smaller rather than bigger steps
                        policy_loss = -T.min(full_loss, clipped_loss).mean()
                        
                    self.scaler.scale(policy_loss + self.entropy_coefficient*new_logits.entropy()).backward()
                    self.scaler.step(self.policy_optim)
                    self.scaler.update()
                
                    ## add KL divergence check to prevent the policy from changing too much
                    ## monte-carlo estimate, not exact
                    kl_div = (old_log_probs-new_log_probs).mean()
                    kl_divs.append(kl_div)
                    
                    if kl_div >= self.target_kl_div:
                        print('KL Divergence exceeded')
                        break
                
                    self.global_training += 1
                
            return True, policy_loss, kl_divs
        
        ## if user does not have a gpu/it doesnt work for them     
        except Exception:
            try:
                kl_divs = []
                for _ in range(self.policy_train_iters):   
                    self.policy_optim.zero_grad()
                    
                    ## following the surrogate loss formula
                    new_logits = self.actor.layers(obs)
                    new_logits = Categorical(logits=new_logits)
                    new_log_probs = new_logits.log_prob(acts)
                    
                    ## getting the ratio of the new policy divided by the old policy
                    policy_ratio = T.exp(new_log_probs - old_log_probs)
                    
                    ## clip the ratio in between 0.8 and 1.2
                    clipped_ratio = policy_ratio.clamp(
                        1-self.ppo_clip_val, 1+self.ppo_clip_val
                    )
                    
                    clipped_loss = clipped_ratio*gaes
                    full_loss = policy_ratio*gaes
                    
                    ## get the smaller of the 2 values - ppo wants to 
                    ## make smaller rather than bigger steps
                    policy_loss = -T.min(full_loss, clipped_loss).mean()
                    
                    policy_loss.backward()
                    self.policy_optim.step()
                    
                    ## add KL divergence check to prevent the policy from
                    ## changing too much - which could ruin the model
                    ## monte-carlo estimate, not exact
                    kl_div = (old_log_probs-new_log_probs).mean()
                    kl_divs.append(kl_div)
                    
                    if kl_div >= self.target_kl_div:
                        print('KL Divergence exceeded')
                        break
                return True, policy_loss, kl_div
            except Exception:
                return False, 0, None
    
    def train_value(self, obs, returns):
        try:
            self.value_optim.zero_grad()
        
            with T.cuda.amp.autocast():
                values = self.critic.layers(obs)
                ## L2 loss
                value_loss = 0.5*(returns - values)**2
                value_loss = value_loss.mean()
            
            self.scaler.scale(value_loss).backward()
            self.scaler.step(self.value_optim)
            self.scaler.update()
            return True, value_loss
            
        except Exception:
            try:
                self.value_optim.zero_grad()
            
                values = self.critic.layers(obs)
                ## L2 loss
                value_loss = 0.5*(returns - values)**2
                value_loss = value_loss.mean()
                
                value_loss.backward()
                self.value_optim.step()
                return True, value_loss
            
            except Exception:
                return False, 0
        
    def save_checkpoint(self):
        T.save({'agent_state_dict': self.agent.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'optim_state_dict': self.optim.state_dict(),},
                self.checkpoint_file_best, _use_new_zipfile_serialization=False)
        
    def routine_save(self):
        T.save({'agent_state_dict': self.agent.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'optim_state_dict': self.optim.state_dict(),},
                self.checkpoint_file_current, _use_new_zipfile_serialization=False)
                
    def load_checkpoint(self):
        ckpt = T.load(self.checkpoint_file_old)
        self.agent.load_state_dict(ckpt['agent_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict']) 
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        

            