import torch
import cv2
from mss import mss
import numpy as np
import control as c
# import os


class Env:
    def __init__(self, height, width, frame_time):
        self.height = height
        self.width = width
        self.frame_time = frame_time
        self.observation_space = (283, 430)
        self.sct = mss()
        self.sct.compression_level = 9
        # self.screen_size = (410, 320, 900, 750)  # hidden (1280x1024x24 middle)
        # self.screen_size = (100, 100, 570, 430)  # one monitor
        self.screen_size = (50, 50, 500, 400)  # one monitor
        self.lower_red = np.array([0, 200, 120])
        self.upper_red = np.array([10, 255, 150])
        self.lower_yellow = np.array([20, 120, 100])
        self.upper_yellow = np.array([30, 255, 255])
        # self.lower_red2 = np.array([0, 200, 60])  # white ball
        # self.upper_red2 = np.array([30, 255, 255])  # white ball
        self.start_warmup = 2.8
        self.warmup = 0.9
        self.pole = 0
        self.ground = 0
        self.threshold = 0.95
        self.polepoint = cv2.imread('img/polepoint/polepoint.png', 0)

    def preprocess_img(self, resize=True, save_dir=False):
        raw = self.sct.grab(self.screen_size)
        img = np.array(raw)
        img = cv2.resize(img, (450, 350))
        if resize:
            img = img[self.ground - 180: self.ground + 100, self.pole - 215:self.pole + 215]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_red = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
        img_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
        self.mask = img_red + img_yellow
        cv2.imshow('img', self.mask)
        cv2.waitKey(1)
        del img
        del img_hsv
        if resize:
            state = cv2.resize(self.mask, (self.width, self.height))
            if save_dir:
                cv2.imwrite(save_dir, state)
            state = state / float(255) * float(2) - 1
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            return state

    def get_standard(self, first_=False, set_=False):
        if first_:
            self.preprocess_img(resize=False)
            gameset_match = cv2.matchTemplate(self.mask, self.polepoint, eval('cv2.TM_CCOEFF_NORMED'))
            if np.max(gameset_match) > self.threshold:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gameset_match)
                top_left = max_loc
                w, h = self.polepoint.shape[::-1]
                x, y = (top_left[0] + w // 2, top_left[1] + h // 2)
                self.pole = x
                self.ground = y
                return True
            else:
                return False

        else:
            self.preprocess_img(resize=True)
            gameset_match = cv2.matchTemplate(self.mask, self.polepoint, eval('cv2.TM_CCOEFF_NORMED'))
            if np.max(gameset_match) > self.threshold:
                return True
            else:
                return False

    def set_standard(self):
        ready = False
        print("Set standard")
        while not ready:
            ready = self.get_standard(first_=True)
            c.p(self.frame_time)
        print("Ready")

    def restart(self):
        start1 = False
        start2 = False
        restart = False
        wait = 0
        while not start1 or start2 or not restart:
            start1 = start2
            start2 = self.check_start()
            restart = self.get_standard()
            wait += 1
            if wait > 500:
                return True
        return False

    def check_start(self):
        return np.sum(self.mask) == 0

    def check_end(self):
        jump = self.mask[:220, 215:].sum() > 60000

        if np.sum(self.mask[-2:, :]) > 1000:
            if np.sum(self.mask[-2:, :215]) > np.sum(self.mask[-2:, 215:]):
                self.win = True
                # if save_dir:
                #     new_dir = save_dir[:-6] + '-1.png'
                #     os.rename(save_dir, new_dir)
                return True, False
            else:
                self.win = False
                # if save_dir:
                #     new_dir = save_dir[:-6] + '-2.png'
                #     os.rename(save_dir, new_dir)
                return True, False
        else:
            return False, jump

    def restart_set(self):
        start1 = False
        start2 = False
        restart = False
        while not start1 or start2 or not restart:
            start1 = start2
            start2 = self.check_start()
            restart = self.get_standard()
            print("Start new set")
            c.release()
            c.p(self.frame_time)
