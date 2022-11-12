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
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PPO_resnet18 import ActorCriticNetwork, PPOTrainer
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
        
        ## save models if the current model is doing well
        global best_score, score, ppo
        if (best_score == None) or (score > best_score):
            ppo.save_checkpoint()
            
        self.error_msg.setText("Model saved.")
        QApplication.processEvents()
        
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
        
        with open('C:\Program Files (x86)\StreamCompanion\Files\\name.txt') as f:
            try:
                name = f.read()
            except Exception: 
                name = ""
        
        if name != " ":
            self.error_msg.show()
            self.error_msg.setText("Error: Please log out of your osu! account")
            QApplication.processEvents()
            return
            
        pdi.FAILSAFE = False ## prevent pdi from adding a delay after inputs
        pdi.PAUSE = 0 
        NUM_EPOCHS = self.epochs.value() ## number of epochs of optimization steps
        ALPHA = self.learning_rate.value() ## learning rate
        NUM_GAMES = self.iterations.value() ## number of times to replay a map
        CHECKPOINT = self.load_pretrained.isChecked() ## true if loading from a chkpt
        NUM_ACTIONS = 12 ## number of total actions the agent can make
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        
        agent = ActorCriticNetwork(n_actions=NUM_ACTIONS)
        agent = agent.to(DEVICE)
        
        ppo = PPOTrainer(
            agent, 
            policy_lr = ALPHA,
            value_lr = 1e-3,
            target_kl_div = 0.02,
            policy_train_iters = NUM_EPOCHS,
            value_train_iters = NUM_EPOCHS
        )
        
        if CHECKPOINT:
            try:
                ppo.load_checkpoint()
            except Exception:
                self.error_msg.setText("Error: Models not found.")
                self.error_msg.show()
                QApplication.processEvents()
                
        ## store the highest reward of the bot - shows improvement
        best_score = None
        score_history = []
        
        ## storing steps for when to update 
        learn_iters = 0
        n_steps = 0
        
        ## kill the program if delete key is pressed
        keyboard.add_hotkey('delete', lambda: self.force_quit(best_score, reward, ppo), suppress=True)
        
        ## finds most optimal cnn algo
        try:
            torch.backends.cudnn.benchmark = True     
        except Exception:
            pass
        
        train = not self.test_mode.isChecked()
        
        if train == False:
            agent.eval()
        
        total_steps = 0
        
        for i in range(NUM_GAMES):   
            ## reset the environment and scores
            self.restart(train=train)
            reward = 0
            
            ## video mode is True to override auto filtering
            camera.start(target_fps=100, video_mode=True)
            
            ## getting first frame as input
            observation = self.initialize()
        
            ## do a full rollout using the old policy
            train_data, reward, n_steps = self.rollout(model=agent, obs=observation, train=train)
            
            ## if playing only is on, don't train model
            if train == True:
                score_history.append(reward)
            
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
                trainp = ppo.train_policy(obs, acts, act_log_probs, gaes)  
                if trainp == False:
                    self.error_msg.setText("Error: Policy training failed.")
                    self.error_msg.show()
                    QApplication.processEvents()
                else:
                    self.error_msg.setText("Trained policy.")
                    self.error_msg.show()
                    QApplication.processEvents()
                    
                trainv = ppo.train_value(obs, returns)        
                if trainv == False:
                    self.error_msg.setText("Error: Value model training failed.")
                    self.error_msg.show()
                    QApplication.processEvents()
                else:
                    self.error_msg.setText("Trained value model.")
                    self.error_msg.show()
                    QApplication.processEvents()
                
                learn_iters += NUM_EPOCHS
                
                ## if current score is better than past scores, save the model
                score_history.append(reward)
            
                if (best_score == None) or (reward > best_score):
                    self.error_msg.show()
                    self.error_msg.setText("Performance improved. Updating model.")
                    QApplication.processEvents()
                    best_score = reward
                    ppo.save_checkpoint()

            ## summarize performance
            total_steps += n_steps
            self.episode.setText(f"Episode: {i+1}") 
            self.score.setText(f"Score: {reward}")
            self.time_steps.setText(f"Time Steps: {total_steps}")
            self.learn_steps.setText(f"Learning Steps: {learn_iters}")
            self.progress_bar.setValue(i+1)
            QApplication.processEvents()
            
            
    ## stop the program
    def force_quit(self, best_score, score, ppo):   
        ## pause the map to prevent score from changing   
        camera.release()     
        pdi.press('escape', _pause=False)
        
        ## save models if the current model is doing well
        if (best_score == None) or (score > best_score):
            ppo.save_checkpoint()
        
        print("Model saved.")
        self.close()


    ## take a image of the window, process it and return it as a (244, 244, 1) tensor
    def screenshot(self):
        ## get the image of the window
        frame = camera.get_latest_frame()

        ## downsample current screen, turn it into a square
        #im = cv2.resize(frame, dsize=(224,224))
        im = cv2.resize(frame, dsize=(112,112))
        
        ## we need to copy the grayscale tensor 3 times bc resnet takes 3x224x224 input
        ## take the image and turn it into a tensor
        #tensor = torchvision.transforms.ToTensor()(np.repeat(im[np.newaxis,:,:], 3, axis=0))
        tensor = torchvision.transforms.ToTensor()(im).expand(3,-1,-1)
        tensor = tensor.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return tensor


    ## one timestep - do action sent from model
    def step(self, action):
        ## do the action sent from model
        if action == 0:
            pdi.leftClick(_pause=False)
        elif action == 1:
            pdi.mouseDown(_pause=False)
        elif action == 2:
            pdi.mouseUp(_pause=False)
        elif action == 3:
            pdi.moveRel(50, 0, _pause=False)
        elif action == 4:
            pdi.moveRel(35, 35, _pause=False)
        elif action == 5:
            pdi.moveRel(-50, 0, _pause=False)
        elif action == 6:
            pdi.moveRel(-35, 35, _pause=False)
        elif action == 7:
            pdi.moveRel(0, 50, _pause=False)
        elif action == 8:
            pdi.moveRel(-35, -35, _pause=False)
        elif action == 9:
            pdi.moveRel(0, -50, _pause=False)
        elif action == 10:
            pdi.moveRel(35, -35, _pause=False)
        elif action == 11:
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
                
        with open('C:\Program Files (x86)\StreamCompanion\Files\combo.txt') as f:
            try:
                combo = int(f.read())
            except Exception: 
                combo = 0
                
        with open('C:\Program Files (x86)\StreamCompanion\Files\misses.txt') as f:
            try:
                misses = int(f.read())
            except Exception: 
                misses = 0   
        
        ## create reward based off stats
        reward = ((perfects*3+goods*2+bads-5*misses)*((100+combo)/100))/100
        
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
            time.sleep(0.5)
            pdi.keyDown("right")
            time.sleep(0.5)
            pdi.keyUp("right")
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


    def discount_rewards(self, rewards, gamma=0.99):
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
            
        return np.array(gaes[::-1])
        
        
    def rollout(self, model, obs, train=True):
        ## observation, action, reward, value, act_log_probs
        train_data = [[],[],[],[],[]]
        
        ep_reward = 0
        n_steps = 0
        done = False
                
        while not done:
            try:
                n_steps += 1

                logits, val = model(obs)
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
                    
                obs = next_obs
                ep_reward += reward
                
                ## if the map is finished, stop and restart
                if done:
                    camera.stop()
                    break 
            
            except Exception as e:
                self.error_msg.show()
                self.error_msg.setText("Error")
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
    