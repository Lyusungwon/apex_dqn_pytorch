import argparse
from environment import Env
import control as c
from model import DQN
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from collections import deque
import random
# import numpy as np
from tensorboardX import SummaryWriter
import os
import gc

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=1000000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--simnum', type=int, default=0, metavar='N')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--save-data', action='store_true', default=False)
parser.add_argument('--device', type=str, default="cpu", metavar='N')
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/alphachu/', metavar='N', help='log directory')
parser.add_argument('--data-directory', type=str, default='/home/sungwonlyu/data/alphachu/', metavar='N', help='data directory')
# parser.add_argument('--log-directory', type=str, default='/Users/SungwonLyu/experiment/alphachu/', metavar='N', help='log directory')
# parser.add_argument('--data-directory', type=str, default='/Users/SungwonLyu/data/alphachu/', metavar='N', help='data directory')
parser.add_argument('--history_size', type=int, default=4, metavar='N')
parser.add_argument('--width', type=int, default=129, metavar='N')
parser.add_argument('--height', type=int, default=84, metavar='N')
parser.add_argument('--hidden-size', type=int, default=32, metavar='N')
parser.add_argument('--epsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--wepsilon', type=float, default=0.9, metavar='N')
parser.add_argument('--frame-time', type=float, default=0.2, metavar='N')
parser.add_argument('--reward', type=float, default=1, metavar='N')
parser.add_argument('--replay-size', type=int, default=3000, metavar='N')
args = parser.parse_args()
torch.manual_seed(args.seed)


class Actor:
    def __init__(self):
        if args.device != 'cpu':
            torch.cuda.set_device(int(args.device))
            self.device = torch.device('cuda:{}'.format(int(args.device)))
        else:
            self.device = torch.device('cpu')

        self.simnum = args.simnum
        self.history_size = args.history_size
        self.height = args.height
        self.width = args.width
        self.hidden_size = args.hidden_size
        if args.test:
            args.epsilon = 0
            args.wepsilon = 0
        self.epsilon = args.epsilon
        self.log = args.log_directory + args.load_model + '/'
        self.writer = SummaryWriter(self.log + str(self.simnum) + '/')

        self.dis = 0.99
        self.win = False
        self.jump = False
        self.ground_key_dict = {0: c.stay,
                                1: c.left,
                                2: c.right,
                                3: c.up,
                                4: c.left_p,
                                5: c.right_p}
        self.jump_key_dict = {0: c.stay,
                              1: c.left_p,
                              2: c.right_p,
                              3: c.up_p,
                              4: c.p,
                              5: c.down_p}
        self.key_dict = self.ground_key_dict
        self.action_size = len(self.key_dict)
        self.replay_memory = deque(maxlen=args.replay_size)
        self.priority = deque(maxlen=args.replay_size)
        self.mainDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.start_epoch = self.load_checkpoint()

    def save_checkpoint(self, idx):
        checkpoint = {'simnum': self.simnum,
                      'epoch': idx + 1}
        torch.save(checkpoint, self.log + 'checkpoint{}.pt'.format(self.simnum))
        print('Actor {}: Checkpoint saved in '.format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))

    def load_checkpoint(self):
        if os.path.isfile(self.log + 'checkpoint{}.pt'.format(self.simnum)):
            checkpoint = torch.load(self.log + 'checkpoint{}.pt'.format(self.simnum))
            self.simnum = checkpoint['simnum']
            print("Actor {}: loaded checkpoint ".format(self.simnum), '(epoch {})'.format(checkpoint['epoch']), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return checkpoint['epoch']
        else:
            print("Actor {}: no checkpoint found at ".format(self.simnum), self.log + 'checkpoint{}.pt'.format(self.simnum))
            return args.start_epoch

    def save_memory(self):
        if os.path.isfile(self.log + 'memory.pt'):
            try:
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
            except:
                time.sleep(10)
                memory = torch.load(self.log + 'memory{}.pt'.format(self.simnum))
                memory['replay_memory'].extend(self.replay_memory)
                memory['priority'].extend(self.priority)
                torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
                self.replay_memory.clear()
                self.priority.clear()
        else:
            memory = {'replay_memory': self.replay_memory,
                      'priority': self.priority}
            torch.save(memory, self.log + 'memory{}.pt'.format(self.simnum))
            self.replay_memory.clear()
            self.priority.clear()

        print('Actor {}: Memory saved in '.format(self.simnum), self.log + 'memory{}.pt'.format(self.simnum))

    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            if args.device == 'cpu':
                model_dict = torch.load(self.log + 'model.pt', map_location=lambda storage, loc: storage)
            else:
                model_dict = torch.load(self.log + 'model.pt')
            self.mainDQN.load_state_dict(model_dict['state_dict'])
            print('Actor {}: Model loaded from '.format(self.simnum), self.log + 'model.pt')

        else:
            print("Actor {}: no model found at '{}'".format(self.simnum, self.log + 'model.pt'))

    def history_init(self):
        history = torch.zeros([1, self.history_size, self.height, self.width])
        return history

    def update_history(self, history, state):
        history = torch.cat([state, history[:, :self.history_size - 1]], 1)
        return history

    def select_action(self, history):
        self.mainDQN.eval()
        history = history.to(self.device)
        qval = self.mainDQN(history)
        self.maxv, action = torch.max(qval, 1)
        sample = random.random()
        if not self.win:
            self.epsilon = args.epsilon
        else:
            self.epsilon = args.wepsilon
        if sample > self.epsilon:
            self.random = False
            action = action.item()
        else:
            self.random = True
            action = random.randrange(self.action_size)
        return action

    def control(self, jump):
        if not jump:
            self.key_dict = self.ground_key_dict
        elif jump:
            self.key_dict = self.jump_key_dict

    def main(self):
        c.release()
        self.load_model()
        env.set_standard()
        total_reward = 0
        set_end = False
        for idx in range(self.start_epoch, args.epochs + 1):
            reward = self.round(idx, set_end)
            self.writer.add_scalar('reward', reward, idx)
            total_reward += reward
            set_end = env.restart()
            if set_end:
                self.writer.add_scalar('total_reward', total_reward, idx)
                total_reward = 0
                self.win = False
                if not args.test:
                    self.save_memory()
                    self.load_model()
                    self.save_checkpoint(idx)
                env.restart_set()
        self.writer.close()

    def round(self, round_num, set_end):
        print("Round {} Start".format(round_num))
        if not set_end:
            time.sleep(env.warmup)
        else:
            time.sleep(env.start_warmup)
        history = self.history_init()
        action = 0
        next_action = 0
        frame = 0
        reward = 0
        estimate = 0
        end = False
        maxv = torch.zeros(0).to(self.device)
        actions = torch.zeros(0).to(self.device)
        start_time = time.time()
        while not end:
            round_time = time.time() - start_time
            sleep_time = args.frame_time - (round_time % args.frame_time)
            time.sleep(sleep_time)
            start_time = time.time()
            if round_time + sleep_time > args.frame_time:
                raise ValueError('Timing error')
            # print(round_time, sleep_time, round_time + sleep_time)
            if args.save_data:
                save_dir = args.data_directory + str(args.time_stamp) + '-' + str(round_num) + '-' + str(frame) + '-' + str(action) + '.png'
            else:
                save_dir = None
            state = env.preprocess_img(save_dir=save_dir)
            next_history = self.update_history(history, state)
            end, jump = env.check_end()
            if not end:
                next_action = self.select_action(next_history)
                self.control(jump)
                print(self.key_dict[next_action])
                self.key_dict[next_action](args.frame_time)
                if not self.random:
                    maxv = torch.cat([maxv, self.maxv])
                    actions = torch.cat([actions, torch.FloatTensor([action]).to(self.device)])
                frame += 1
                priority = abs(self.dis * self.maxv.item() - estimate)
                estimate = self.maxv.item()
            else:
                c.release()
                if env.win:
                    reward = args.reward
                else:
                    reward = - args.reward
                priority = abs(reward - estimate)
            if not args.test:
                self.replay_memory.append((history, action, reward, next_history, end))
                self.priority.append(priority)
            history = next_history
            action = next_action
            if frame > 2000:
                raise ValueError('Loop bug')
        if maxv.size()[0] > 0:
            self.writer.add_scalar('maxv', maxv.mean(), round_num)
        if actions.size()[0] > 0:
            self.writer.add_scalar('action', actions.mean(), round_num)
        self.writer.add_scalar('epsilon', self.epsilon, round_num)
        self.writer.add_scalar('frame', frame, round_num)
        gc.collect()
        if env.win:
            print("Round {} Win: reward:{}, frame:{}".format(round_num, reward, frame))
            self.win = True
        else:
            print("Round {} Lose: reward:{}, frame:{}".format(round_num, reward, frame))
            self.win = False
        return reward


if __name__ == "__main__":
    env = Env(args.height, args.width, args.frame_time)
    actor = Actor()
    actor.main()
