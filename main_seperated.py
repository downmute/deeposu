import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torchvision
import dxcam
import pydirectinput as pdi
import win32gui, win32api
import time
import sys
import cv2
import keyboard
from PyQt5.QtWidgets import QMainWindow, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLabel, QProgressBar, QAction, QApplication
from PyQt5 import QtGui
from torch.utils.tensorboard import SummaryWriter
import pickle
writer = SummaryWriter('runs/v4_retrained')

#from PPO_resnet18 import ActorCriticNetwork, PPOTrainer
from ppo_seperate import Agent, PPOTrainer
from app import Ui_MainWindow


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setWindowIcon(QtGui.QIcon('deeposu.ico'))
        self.setWindowTitle("deeposu")
        self.setFixedSize(420, 350)
        
        self.epochs = self.findChild(QSpinBox, "epochs")
        self.iterations = self.findChild(QSpinBox, "iterations")
        self.learning_rate = self.findChild(QDoubleSpinBox, "learning_rate")
        self.load_pretrained = self.findChild(QCheckBox, "load_pretrained")
        self.test_mode = self.findChild(QCheckBox, "eval_mode")
        self.run = self.findChild(QPushButton, "submit")
        self.episode = self.findChild(QLabel, "episode")
        self.score = self.findChild(QLabel, "score")
        self.time_steps = self.findChild(QLabel, "time_steps")
        self.learn_steps = self.findChild(QLabel, "learn_steps")
        self.progress_bar = self.findChild(QProgressBar, "progressBar")
        self.error_msg = self.findChild(QLabel, "error_msg")
        self.progress_bar.setMaximum(self.iterations.value())
        
        self.exit = QAction("Exit Application", shortcut= QtGui.QKeySequence("Ctrl+Q"), triggered=lambda:self.exit_app)
        
        self.run.clicked.connect(self.clicked)
        
        self.show()
        self.error_msg.setText("")
        self.error_msg.hide()
        
    def exit_app(self):
        ## pause the map to prevent score from changing   
        camera.release()     
        pdi.press('escape', _pause=False)       
        self.close()
    
    ## when run is clicked, start the training
    def clicked(self):
        self.progress_bar.setMaximum(self.iterations.value())
        self.progress_bar.setValue(0)
        self.error_msg.setText("")
        QApplication.processEvents()
        ## ensure user has streamcompanion open so reward is calculated
        tabs = []
        def findtab(hwnd, _):
            if "StreamCompanion" in (win32gui.GetWindowText(hwnd)):
                tabs.append(hwnd)
        win32gui.EnumWindows(findtab, None)
        if tabs == []:
            self.error_msg.show()
            self.error_msg.setText("Error: StreamCompanion is not open - cannot get reward")
            
            return
        
        with open('C:\Program Files (x86)\StreamCompanion\Files\\connected.txt') as f:
            try:
                name = f.read()
            except Exception: 
                name = ""
        
        if name != ' ':
            self.error_msg.show()
            self.error_msg.setText("Error: Please log out of your osu! account")
            QApplication.processEvents()
            return
            
        pdi.FAILSAFE = False ## prevent pdi from adding a delay after inputs
        pdi.PAUSE = 0 
        NUM_EPOCHS = self.epochs.value() ## number of epochs of optimization steps
        ALPHA = self.learning_rate.value() ## learning rate
        NUM_GAMES = self.iterations.value() ## number of times to replay maps
        CHECKPOINT = self.load_pretrained.isChecked() ## true if loading from a chkpt
        NUM_ACTIONS = 10 ## number of total actions the agent can make
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        
        agent = Agent(n_actions=NUM_ACTIONS)
        agent = agent.to(DEVICE)
        
        ppo = PPOTrainer(
            agent, 
            lr = ALPHA,
            target_kl_div = 0.02,
            train_iters = NUM_EPOCHS,
        )
        
        if CHECKPOINT:
            try:
                ppo.load_checkpoint()
            except Exception:
                self.error_msg.setText("Error: Models not found.")
                self.error_msg.show()
                QApplication.processEvents()
                
        ## store the highest reward of the bot - shows improvement
        '''best_score = agent.get_score()
        print(f'Best score: {best_score}')
        try:
            best_score = float(best_score)
        except:
            best_score = None'''
        
        try:
            with open('./highscore.dat') as f:
                best_score = pickle.load(f)
        except:
            print('Could not find highscore file')
            best_score = -1000000
        print(f'Best score: {best_score}')
        
        ## storing steps for when to update 
        learn_iters = 0
        n_steps = 0
        
        ## kill the program if delete key is pressed
        keyboard.add_hotkey('delete', lambda: self.force_quit(), suppress=True)
        
        ## finds most optimal cnn algo
        try:
            torch.backends.cudnn.benchmark = True     
        except Exception:
            pass
        
        train = not self.test_mode.isChecked()
        
        if train == False:
            agent.eval()
        
        total_steps = 0   
        rewards_history = [None,None,None,None,None,None,None,None,None,None]   
        
        for i in range(NUM_GAMES):   
            ## reset the environment and scores
            time.sleep(0.5)
            self.restart(train=train)
            reward = 0
            
            ## video mode is True to override auto filtering
            camera.start(target_fps=100, video_mode=True)
            
            ## getting first frame as input
            observation = self.initialize()
        
            ## do a full rollout using the old policy
            train_data, reward, n_steps = self.rollout(agent=agent, obs=observation, train=train)
            
            ## if playing only is on, don't train model
            if train == True and i != 0:
            
                ## shuffle 
                permute_idxs = np.random.permutation(len(train_data[0]))
                                                  
                ## policy data          
                for data in range(len(train_data)):
                    train_data[data] = np.array(train_data[data], dtype=[('O', np.float32)]).astype(np.float32)
                
                obs = torch.tensor(train_data[0][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)
                acts = torch.tensor(train_data[1][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)
                returns = torch.tensor(train_data[2][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)
                gaes = torch.tensor(train_data[3][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)
                act_log_probs = torch.tensor(train_data[4][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)
                
                ## ppo trainer automatically trains multiple epochs
                trained, kl_divs, p_loss, v_loss, t_loss = ppo.train(obs, acts, act_log_probs, gaes, returns, batches=8)  
                if trained == False:
                    self.error_msg.setText("Error: Training failed.")
                    self.error_msg.show()
                    QApplication.processEvents()
                else:
                    self.error_msg.setText("Trained.")
                    self.error_msg.show()
                    QApplication.processEvents()
                    ()
                
                learn_iters += NUM_EPOCHS
                                
                
                ## log data on tensorboard for evaluation
                writer.add_scalar('reward',
                                reward,
                                i)
                
                writer.add_scalar('avg_policy_loss',
                                p_loss,
                                i)
                
                writer.add_scalar('avg_value_loss',
                                v_loss,
                                i)
                
                writer.add_scalar('avg_total_loss',
                                t_loss,
                                i)
                
                for n in range(len(kl_divs)):
                    writer.add_scalar('kl_div',
                                      kl_divs[n],
                                     (i*len(kl_divs))+n)
                    
                
                ## if current score is better than past scores, save the model
                '''best_score = agent.get_score()
                print(f'Best score before: {best_score}')
                try:
                    best_score = float(best_score)
                except:
                    best_score = None
                print(f'Best score after: {best_score}')'''
                
                total_rewards = 0
                minus_count = 0
                rewards_history = rewards_history[1:]
                rewards_history.append(reward)
                for reward in rewards_history:
                    if reward != None:
                        total_rewards += reward
                    else:
                        minus_count += 1
                mean_reward = total_rewards/(len(rewards_history) - minus_count)
                
                if mean_reward > best_score:
                    self.error_msg.show()
                    self.error_msg.setText("Model saved!")
                    QApplication.processEvents()
                    best_score = reward
                    with open('./highscore.dat', 'wb') as f:
                        pickle.dump(best_score, f)
                    #agent.set_score(f'{best_score}')
                    ppo.save_checkpoint()
                    print('Saved Checkpoint')

                
                ## summarize performance
                total_steps += n_steps
                self.episode.setText(f"Episode: {i+1}") 
                self.score.setText(f"Score: {round(reward,5)}")
                self.time_steps.setText(f"Time Steps: {total_steps}")
                self.learn_steps.setText(f"Learning Steps: {learn_iters}")
                self.progress_bar.setValue(i+1)
                QApplication.processEvents()
            
            
    ## stop the program
    def force_quit(self):   
        ## pause the map to prevent score from changing   
        camera.release()     
        pdi.press('escape', _pause=False)
        self.close()


    ## take a image of the window, process it and return it as a (96, 96, 1) tensor
    def screenshot(self):
        ## get the image of the window
        frame = camera.get_latest_frame()

        ## downsample current screen, turn it into a square        
        im = cv2.resize(frame, dsize=(96,96))        
        
        ## take the image and turn it into a tensor
        tensor = torch.unsqueeze(torchvision.transforms.ToTensor()(im), 0)        
        tensor = tensor.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return tensor

    ## formula for reward calculation
    def calc_reward(self, perfects, goods, bads, misses):
        #return round(((perfects+goods*0.5+bads*0.2)*(combo+1)-(misses*50))*0.001, 4)
        return (perfects*1+goods*0.25+bads*0.1-misses*10)*0.00001

    ## one timestep - do action sent from model
    def step(self, action):
        ## do the action sent from model
        if action == 0:
            pdi.mouseDown(_pause=False)
        elif action == 1:
            pdi.moveRel(30, 0, _pause=False)
        elif action == 2:
            pdi.moveRel(42, 42, _pause=False)
        elif action == 3:
            pdi.moveRel(-30, 0, _pause=False)
        elif action == 4:
            pdi.moveRel(-42, 42, _pause=False)
        elif action == 5:
            pdi.moveRel(0, 30, _pause=False)
        elif action == 6:
            pdi.moveRel(42, -42, _pause=False)
        elif action == 7:
            pdi.moveRel(0, -30, _pause=False)
        elif action == 8:
            pdi.moveRel(-42, -42, _pause=False)
        elif action == 9:
            pass
        
        ## see result of action
        observation = self.screenshot()
        
        ## check if the map is finished playing
        with open('C:\Program Files (x86)\StreamCompanion\Files\\finished.txt') as f:
            status = f.read()
            if status == "ResultsScreen":
                finished = True
            else:
                finished = False
        
        ## look at memory to see stats 
        try:
            with open('C:\Program Files (x86)\StreamCompanion\Files\\300.txt') as f:
                try:
                    perfects = int(f.read())
                except Exception: 
                    perfects = 0
                    
            with open('C:\Program Files (x86)\StreamCompanion\Files\\100.txt') as f:
                try:
                    goods = int(f.read())
                except Exception: 
                    goods = 0
                    
            with open('C:\Program Files (x86)\StreamCompanion\Files\\50.txt') as f:
                try:
                    bads = int(f.read())
                except Exception: 
                    bads = 0
                    
            with open('C:\Program Files (x86)\StreamCompanion\Files\misses.txt') as f:
                try:
                    misses = int(f.read())
                except Exception: 
                    misses = 0   
                    
        except Exception:
            print(Exception)
        
        ## create reward based off stats
        reward = self.calc_reward(perfects, goods, bads, misses) 
        
        return observation, reward, finished
            
            
    def initialize(self):
        ## get the coordinates of the center of the screen, move there
        wcenter = int(win32api.GetSystemMetrics(0)/2) 
        hcenter = int(win32api.GetSystemMetrics(1)/2)
        pdi.moveTo(wcenter, hcenter) 
        return self.screenshot() ## return the initial state of gameplay


    def restart(self, train=True):
        if train == True:
            pdi.keyDown('escape')
            time.sleep(0.5)
            pdi.keyUp('escape')
            pdi.keyDown('alt', _pause=False)
            time.sleep(0.25)
            pdi.keyDown('f4', _pause=False)
            time.sleep(0.25)
            pdi.keyUp('alt', _pause=False)
            pdi.keyUp('f4', _pause=False)
            time.sleep(1)
            '''
            pdi.keyDown("right")
            time.sleep(0.5)
            pdi.keyUp("right")
            '''
            pdi.keyDown('enter')
            time.sleep(0.5)
            pdi.keyUp('enter')
            time.sleep(0.5)
            pdi.keyDown('shift')
            pdi.keyDown('`')
            time.sleep(0.5)
            pdi.keyUp('shift')
            pdi.keyUp('`')
            time.sleep(1) 
            pdi.press('space')
            
        else:
            pdi.keyDown('escape')
            time.sleep(0.5)
            pdi.keyUp('escape')
            time.sleep(0.5)
            pdi.keyDown('enter')
            time.sleep(1)
            pdi.keyUp('enter')
            time.sleep(1)
            pdi.keyDown('shift')
            pdi.keyDown('`')
            time.sleep(0.5)
            pdi.keyUp('shift')
            pdi.keyUp('`')
            pdi.press('space')  


    def discount_rewards(self, rewards, gamma=0.98):
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards)-1)):
            new_rewards.append(float(rewards[i]) + gamma*new_rewards[-1])
        return np.array(new_rewards[::-1])


    def calculate_gaes(self, rew, val, gamma=0.99, decay=0.97):
        next_val = np.concatenate([val[1:], [0]])
        deltas = [rew + gamma*next_val - val for rew, val, next_val in zip(rew, val, next_val)] 
        
        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay*gamma*gaes[-1])
        
        ## normalizing the advantages
    
        mean = np.mean(gaes[::-1])
        std_dev = np.std(gaes[::-1])

        normalized_gaes = [(gae-mean)/(std_dev+1e-8) for gae in (gaes[::-1])]
            
        return np.array(normalized_gaes)
        
        
    def rollout(self, agent, obs, train=True):
        ## observation, action, reward, value, act_log_probs
        train_data = [[],[],[],[],[]]
        
        ep_reward = 0
        n_steps = 0
        done = False
                
        while not done:
            try:
                n_steps += 1

                logits, val = agent.get_action_and_value(obs)
    
                act_distribution = Categorical(logits=logits)
                act = act_distribution.sample()
        
                act_log_prob = act_distribution.log_prob(act).item()
                act, val = act.item(), val.item()
                
                next_obs, reward, done = self.step(act)          
                
                if train == True:                                                               
                    for i, item in enumerate((obs, act, reward, val, act_log_prob)):
                        try:
                            item = item.cpu().numpy()
                        except Exception:
                            pass
                        train_data[i].append(item)
                    
                    if n_steps == 1024:
                        camera.stop()
                        pdi.press('escape', _pause=False)
                        break    
                   
                obs = next_obs
                ep_reward += reward     
                
                ## if the map is finished, stop and restart
                if done:
                    camera.stop()
                    break 
            
            except Exception as e:
                self.error_msg.show()
                self.error_msg.setText("Error: Training failed")
                print(e)
                QApplication.processEvents()
        
        if train == True:            
            train_data[3] = self.calculate_gaes(train_data[2], train_data[3])
                
        return train_data, ep_reward, n_steps
    


if __name__ == "__main__":   
    try:
        app = QApplication(sys.argv)
        UIWindow = UI()
        tabs = []  
        def findtab(hwnd, _):
            if "osu!" in (win32gui.GetWindowText(hwnd)):
                tabs.append(hwnd)
        win32gui.EnumWindows(findtab, None)
        
        x1, y1, x2, y2 = win32gui.GetWindowRect(tabs[0])
        camera = dxcam.create(region=(x1+5, y1+30, x2-5, y2-5), output_color="GRAY")
    
        app.exec_()
        sys.exit()   
             
    except Exception:
        sys.exit()
    