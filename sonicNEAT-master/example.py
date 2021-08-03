# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import retro
import numpy as np
import cv2
import neat
import pickle

#config files
config = 'config-feedforward' #inx = (int(inx/8))
#config = 'config-feedforward-loop-winner-maybe' #inx = (int(inx/10))

# the best genome from training
best_genome = pickle.load(open('checkpoints/correct_loop.pkl', 'rb'))
#best_genome = pickle.load(open('checkpoints/cheating_winner_RNN.pkl', 'rb'))

# same config as the config from training
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config)

# create the neural network 
net = neat.nn.recurrent.RecurrentNetwork.create(best_genome, config)
#net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# play Sonic
env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
#env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act2")

state = env.reset()
x, y, _ = env.observation_space.shape

# may need changed depending on the config file used
x = int(x/8)
y = int(y/8)

done = False
while not done:
    env.render()
    
    state = cv2.resize(state, (x, y))
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = np.reshape(state, (x,y))

    imgarray = np.ndarray.flatten(state)

    nnOutput = net.activate(imgarray)
    state, rew, done, info = env.step(nnOutput)
    if info['level_end_bonus']: done = True
    
    if done:
        env.reset()
        env.close()

    