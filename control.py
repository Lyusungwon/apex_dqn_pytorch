from pynput.keyboard import Key, Controller
import time

keyboard = Controller()


def release():
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.release(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.enter)

def stay(press_time):
    keyboard.release(Key.right)
    keyboard.release(Key.left)

def left(press_time):
    keyboard.release(Key.right)
    keyboard.press(Key.left)

def right(press_time):
    keyboard.release(Key.left)
    keyboard.press(Key.right)

def up(press_time):
    keyboard.press(Key.up)
    time.sleep(press_time / 10)
    keyboard.release(Key.up)

def p(press_time):
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.press(Key.enter)
    time.sleep(press_time / 10)
    keyboard.release(Key.enter)

def left_p(press_time):
    keyboard.release(Key.right)
    keyboard.press(Key.left)
    keyboard.press(Key.enter)
    time.sleep(press_time / 10)
    keyboard.release(Key.enter)

def right_p(press_time):
    keyboard.release(Key.left)
    keyboard.press(Key.right)
    keyboard.press(Key.enter)
    time.sleep(press_time / 10)
    keyboard.release(Key.enter)

def up_p(press_time):
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.press(Key.up)
    keyboard.press(Key.enter)
    time.sleep(press_time / 10)
    keyboard.release(Key.enter)
    keyboard.release(Key.up)

def down_p(press_time):
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.press(Key.down)
    keyboard.press(Key.enter)
    time.sleep(press_time / 10)
    keyboard.release(Key.enter)
    keyboard.release(Key.down)




# def release():
#     keyboard.release(Key.left)
#     keyboard.release(Key.right)
#     keyboard.release(Key.up)
#     keyboard.release(Key.down)
#     keyboard.release(Key.enter)

# def stay(press_time):
#     keyboard.release(Key.left)
#     keyboard.release(Key.right)

# def left(press_time):
#     keyboard.release(Key.right)
#     keyboard.press(Key.left)

# def right(press_time):
#     keyboard.release(Key.left)
#     keyboard.press(Key.right)

# def up(press_time):
#     keyboard.press(Key.up)
#     time.sleep(press_time/2)
#     keyboard.release(Key.up)

# def p(press_time):
#     keyboard.press(Key.enter)
#     time.sleep(press_time/2)
#     keyboard.release(Key.enter)

# def p_down(press_time):
#     release()
#     keyboard.press(Key.down)
#     keyboard.press(Key.enter)
#     time.sleep(press_time/2)
#     keyboard.release(Key.down)
#     keyboard.release(Key.enter)
