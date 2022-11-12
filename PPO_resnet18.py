import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torchvision


class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions):
        super(ActorCriticNetwork, self).__init__()
                
        resnet18 = torchvision.models.resnet18(weights="DEFAULT")
        
        for param in resnet18.parameters():
            param.requires_grad = False
        
        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.shared_layers = nn.Sequential(*self.features)
        
        in_features = resnet18.fc.in_features
        
        self.policy_layers = nn.Linear(in_features, n_actions)
        self.value_layers = nn.Linear(in_features, 1)
    
    def value(self, obs):
        z = self.shared_layers(obs)
        z = z.view(-1, 512)
        value = self.value_layers(z)
        return value
    
    def policy(self, obs):
        z = self.shared_layers(obs)
        z = z.view(-1, 512)
        policy = self.policy_layers(z)
        return policy
        
    def forward(self, state):
        if len(state.size()) == 3:
            state = T.unsqueeze(state, dim=0)
        state = self.shared_layers(state)
        state = state.view(-1, 512)
        policy_logits = self.policy_layers(state)
        value = self.value_layers(state)     

        ## creates a distribution of prob of actions to sample from
        return policy_logits, value       
   
   
class PPOTrainer():
    def __init__(self, 
                 actor_critic,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 policy_train_iters=100,
                 value_train_iters=100,
                 policy_lr=3e-4,
                 value_lr=1e-2,
                 chkpt_dir="./models/"):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.policy_train_iters = policy_train_iters
        self.value_train_iters = value_train_iters
        self.checkpoint_file = chkpt_dir + "ppo_resnet18.pth"
        
        policy_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.policy_layers.parameters()) 
        self.policy_optim = optim.AdamW(policy_params, lr=policy_lr)
        
        value_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.value_layers.parameters())
        self.value_optim = optim.AdamW(value_params, lr=value_lr)
        
        self.scaler = T.cuda.amp.GradScaler()
        
    def train_policy(self, obs, acts, old_log_probs, gaes):
        try:
            for _ in range(self.policy_train_iters):   
                self.policy_optim.zero_grad()
                
                ## following the surrogate loss formula
                with T.cuda.amp.autocast():
                    new_logits = self.ac.policy(obs)
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
                    
                self.scaler.scale(policy_loss).backward()
                self.scaler.step(self.policy_optim)
                self.scaler.update()
                
                ## add KL divergence check to prevent the policy from
                ## changing too much - which could ruin the model
                ## monte-carlo estimate, not exact
                kl_div = (old_log_probs-new_log_probs).mean()
                if kl_div >= self.target_kl_div:
                    break
            return True
        
        ## if user does not have a gpu/it doesnt work for them     
        except Exception:
            try:
                for _ in range(self.policy_train_iters):   
                    self.policy_optim.zero_grad()
                    
                    ## following the surrogate loss formula
                    new_logits = self.ac.policy(obs)
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
                    if kl_div >= self.target_kl_div:
                        break
                return True
            except Exception:
                return False
    
    def train_value(self, obs, returns):
        try:
            self.value_optim.zero_grad()
        
            with T.cuda.amp.autocast():
                values = self.ac.value(obs)
                ## L2 loss
                value_loss = (returns - values)**2
                value_loss = value_loss.mean()
            
            self.scaler.scale(value_loss).backward()
            self.scaler.step(self.value_optim)
            self.scaler.update()
            return True
            
        except Exception:
            try:
                self.value_optim.zero_grad()
            
                values = self.ac.value(obs)
                ## L2 loss
                value_loss = (returns - values)**2
                value_loss = value_loss.mean()
                
                value_loss.backward()
                self.value_optim.step()
                return True
            
            except Exception:
                return False
        
    def save_checkpoint(self):
        T.save({'model_state_dict': self.ac.state_dict(),
                'policy_state_dict': self.policy_optim.state_dict(),
               'value_state_dict': self.value_optim.state_dict()},
                self.checkpoint_file, _use_new_zipfile_serialization=False)
                
    def load_checkpoint(self):
        ckpt = T.load(self.checkpoint_file)
        self.ac.load_state_dict(ckpt['model_state_dict'])
        self.policy_optim.load_state_dict(ckpt['policy_state_dict']) 
        self.value_optim.load_state_dict(ckpt['value_state_dict'])
        

            