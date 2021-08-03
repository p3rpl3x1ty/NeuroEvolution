import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
#env = retro.make(game = "SonicTheHedgehog-Genesis", state = "loop")
imgarray = []
xpos_end = 0

resume = True
restore_file = "neat-checkpoint-31"


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        state = env.reset()

        x, y, _ = env.observation_space.shape

        x = int(x/8)
        y = int(y/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitness_current = 0
        counter = 0       
        current_rings = 0
        current_score = 0
        current_x = 0
        
        done = False
        while not done:           
            env.render()
            
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = cv2.resize(state, (y, x))

            # state = np.reshape(state, (x,y))
            # cv2.imshow('main', state)
            # cv2.waitKey(1)
            
            imgarray = np.ndarray.flatten(state)
            nnOutput = net.activate(imgarray)
            
            state, rew, done, info = env.step(nnOutput)
          
            if info['x'] != current_x:
                fitness_current += .01
                current_x = info['x']
            else:
                counter += 1
                
            if info['rings'] != current_rings:
                fitness_current += (info['rings'] - current_rings)
                current_rings = info['rings']
                
            if info['score'] != current_score:
                fitness_current += (info['score'] - current_score)
                current_score = info['score']
                
            if info['level_end_bonus']:
                fitness_current += 1000000
            
            if counter > 1200:
                done = True
                
            if done:
                print(genome_id, fitness_current)
                genome.fitness = fitness_current

##############################################################################
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
if resume == True:
    p = neat.Checkpointer.restore_checkpoint(('checkpoints/' + restore_file))
else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(1, filename_prefix='checkpoints/sonic-checkpoint-'))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
