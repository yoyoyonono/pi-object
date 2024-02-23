'''
Project: Earthrover Robot
Author: Jitesh Saini
Github: https://github.com/jiteshsaini
website: https://helloworld.co.in

'''

import os, time
from roboclaw import Roboclaw

edgetpu=0 # If Coral USB Accelerator connected, then make it '1' otherwise '0'

m1_1 = 8
m1_2 = 11
m2_1 = 14 
m2_2 = 15 
cam_light = 17
headlight_right = 18
headlight_left = 27 
sp_light=9 
address = 0x80
roboclaw = Roboclaw("COM15", 38400)


def init_gpio():
    while not roboclaw.Open():
        pass

def back(value):
    roboclaw.BackwardM1(address,value)
    roboclaw.BackwardM2(address,value)	
    print("moving back!!!!!!")
    
def right(value):
    roboclaw.BackwardM1(address,value)
    roboclaw.ForwardM2(address,value)
    print("right")

def left(value):
    roboclaw.ForwardM1(address,value)
    roboclaw.BackwardM2(address,value)
    print("left")
    
def forward(value):
    roboclaw.ForwardM1(address,value)
    roboclaw.ForwardM2(address,value)
    print("forward")
    
def stop():
    roboclaw.ForwardM1(address,0)
    roboclaw.ForwardM2(address,0)
    roboclaw.BackwardM1(address,0)
    roboclaw.BackwardM2(address,0)
    print("stop")

def speak_tts(text,gender):
    pass
    
def camera_light(state):
    if(state=="ON"):
        print("light on")
    else:
        print("light off")
        
def head_lights(state):
    if(state=="ON"):
        print("light on")
    else:
        print("light off")
        
def red_light(state):
    if(state=="ON"):
        print("light on")
    else:
        print("light off")
    
