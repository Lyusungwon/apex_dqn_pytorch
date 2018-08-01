import argparse
import control as c
from model import DQN
import datetime
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
from collections import deque
from tensorboardX import SummaryWriter
import numpy as np
import os
import gc
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='number of cuda')
parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N',
                    help='time of the run(no modify)')
parser.add_argument('--load-model', type=str, default='000000000000', metavar='N',
                    help='load previous model')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                    help='start-epoch number')
parser.add_argument('--log-directory', type=str, default='/home/sungwonlyu/experiment/alphachu/', metavar='N',
                    help='log directory')
parser.add_argument('--history_size', type=int, default=4, metavar='N')
parser.add_argument('--width', type=int, default=129, metavar='N')
parser.add_argument('--height', type=int, default=84, metavar='N')
parser.add_argument('--hidden-size', type=int, default=32, metavar='N')
parser.add_argument('--action-size', type=int, default=6, metavar='N')
parser.add_argument('--reward', type=int, default=1, metavar='N')
parser.add_argument('--replay-size', type=int, default=30000, metavar='N')
parser.add_argument('--update-cycle', type=int, default=1500, metavar='N')
parser.add_argument('--actor-num', type=int, default=10, metavar='N')
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
args.device = torch.device("cuda:{}".format(args.gpu) if not args.no_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

config_list = [args.batch_size, args.lr, args.history_size,
               args.height, args.width, args.hidden_size,
               args.reward, args.replay_size,
               args.update_cycle, args.actor_num]
config = ""
for i in map(str, config_list):
    config = config + '_' + i
print("Config:", config)


class Learner():
    def __init__(self):
        self.device = args.device
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.history_size = args.history_size
        self.replay_size = args.replay_size
        self.width = args.width
        self.height = args.height
        self.hidden_size = args.hidden_size
        self.action_size = args.action_size
        self.update_cycle = args.update_cycle
        self.log_interval = args.log_interval
        self.actor_num = args.actor_num
        self.alpha = 0.7
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.beta_increment = 1e-6
        self.e = 1e-6
        self.dis = 0.99
        self.start_epoch = 0
        self.mainDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.targetDQN = DQN(self.history_size, self.hidden_size, self.action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.mainDQN.parameters(), lr=args.lr)
        self.replay_memory = deque(maxlen=self.replay_size)
        self.priority = deque(maxlen=self.replay_size)

        if args.load_model != '000000000000':
            self.log = args.log_directory + args.load_model + '/'
            args.time_stamp = args.load_model[:12]
            args.start_epoch = self.load_model()
        self.log = args.log_directory + args.time_stamp + config + '/'
        self.writer = SummaryWriter(self.log)

    def update_target_model(self):
        self.targetDQN.load_state_dict(self.mainDQN.state_dict())

    def save_model(self, train_epoch):
        model_dict = {'state_dict': self.mainDQN.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'train_epoch': train_epoch}
        torch.save(model_dict, self.log + 'model.pt')
        print('Learner: Model saved in ', self.log + 'model.pt')

    def load_model(self):
        if os.path.isfile(self.log + 'model.pt'):
            model_dict = torch.load(self.log + 'model.pt')
            self.mainDQN.load_state_dict(model_dict['state_dict'])
            self.optimizer.load_state_dict(model_dict['optimizer_dict'])
            self.update_target_model()
            self.start_epoch = model_dict['train_epoch']
            print("Learner: Model loaded from {}(epoch:{})".format(self.log + 'model.pt', str(self.start_epoch)))
        else:
            raise "=> Learner: no model found at '{}'".format(self.log + 'model.pt')

    def load_memory(self, simnum):
        if os.path.isfile(self.log + 'memory{}.pt'.format(simnum)):
            try:
                memory_dict = torch.load(self.log + 'memory{}.pt'.format(simnum))
                self.replay_memory.extend(memory_dict['replay_memory'])
                self.priority.extend(memory_dict['priority'])
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                memory_dict['replay_memory'].clear()
                memory_dict['priority'].clear()
                torch.save(memory_dict, self.log + 'memory{}.pt'.format(simnum))
            except:
                time.sleep(10)
                memory_dict = torch.load(self.log + 'memory{}.pt'.format(simnum))
                self.replay_memory.extend(memory_dict['replay_memory'])
                self.priority.extend(memory_dict['priority'])
                print('Memory loaded from ', self.log + 'memory{}.pt'.format(simnum))
                memory_dict['replay_memory'].clear()
                memory_dict['priority'].clear()
                torch.save(memory_dict, self.log + 'memory{}.pt'.format(simnum))
        else:
            print("=> Learner: no memory found at ", self.log + 'memory{}.pt'.format(simnum))

    def sample(self):
        priority = (np.array(self.priority) + self.e) ** self.alpha
        weight = (len(priority) * priority) ** -self.beta
        # weight = map(lambda x: x ** -self.beta, (len(priority) * priority))
        weight /= weight.max()
        self.weight = torch.tensor(weight, dtype=torch.float)
        priority = torch.tensor(priority, dtype=torch.float)
        return torch.utils.data.sampler.WeightedRandomSampler(priority, self.batch_size, replacement=True)

    def main(self):
        train_epoch = self.start_epoch
        self.save_model(train_epoch)
        is_memory = False
        while len(self.replay_memory) < self.batch_size * 100:
            print("Memory not enough")
            for i in range(self.actor_num):
                is_memory = os.path.isfile(self.log + '/memory{}.pt'.format(i))
                if is_memory:
                    self.load_memory(i)
                time.sleep(1)
        while True:
            self.optimizer.zero_grad()
            self.mainDQN.train()
            self.targetDQN.eval()
            x_stack = torch.zeros(0, self.history_size, self.height, self.width).to(self.device)
            y_stack = torch.zeros(0, self.action_size).to(self.device)
            w = []
            self.beta = min(1, self.beta_init + train_epoch * self.beta_increment)
            sample_idx = self.sample()
            for idx in sample_idx:
                history, action, reward, next_history, end = self.replay_memory[idx]
                history = history.to(self.device)
                next_history = next_history.to(self.device)
                Q = self.mainDQN(history)
                if end:
                    tderror = reward - Q[0, action]
                    Q[0, action] = reward
                else:
                    qval = self.mainDQN(next_history)
                    tderror = reward + self.dis * self.targetDQN(next_history)[0, torch.argmax(qval, 1)] - Q[0, action]
                    Q[0, action] = reward + self.dis * self.targetDQN(next_history)[0, torch.argmax(qval, 1)]
                x_stack = torch.cat([x_stack, history.data], 0)
                y_stack = torch.cat([y_stack, Q.data], 0)
                w.append(self.weight[idx])
                self.priority[idx] = tderror.abs().item()
            pred = self.mainDQN(x_stack)
            w = torch.tensor(w, dtype=torch.float, device=self.device)
            loss = torch.dot(F.smooth_l1_loss(pred, y_stack.detach(), reduce=False).sum(1), w.detach())
            loss.backward()
            self.optimizer.step()
            loss /= self.batch_size
            self.writer.add_scalar('loss', loss.item(), train_epoch)
            train_epoch += 1
            gc.collect()
            if train_epoch % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {}'.format(train_epoch, loss.item()))
                self.writer.add_scalar('replay size', len(self.replay_memory), train_epoch)
                if (train_epoch // self.log_interval) % args.actor_num == 0:
                    self.save_model(train_epoch)
                self.load_memory((train_epoch // self.log_interval) % args.actor_num)

            if train_epoch % self.update_cycle == 0:
                self.update_target_model()


if __name__ == "__main__":
    learner = Learner()
    learner.main()
