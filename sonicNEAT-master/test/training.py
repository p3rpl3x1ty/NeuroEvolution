import retro
import numpy as np
import cv2
import neat
import pickle
import memory

env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
#env = retro.make(game = "SonicTheHedgehog-Genesis", state = "loop")

imgarray = []

resume = False
restore_file = "neat-checkpoint-268"

m = memory.memory()
longest_life = 0
def eval_genomes(genomes, config):
    global longest_life
    for genome_id, genome in genomes:
        state = env.reset()
        state, rew,done, info = env.step([0,0,0,0,0,0,0,0,0,0,0,0])
        start = (info['x'], info['y'])

        x, y, _ = env.observation_space.shape

        x = int(x/8)
        y = int(y/8)

        #net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitness_current = 0
        counter = 0     
        life = 0
        current_x = 0
        
        done = False
        while not done:    
            if life > longest_life:
                longest_life = life
            life += 1
            env.render()
            
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = cv2.resize(state, (y, x))

            #state = np.reshape(state, (x,y))
            #cv2.imshow('main', state)
            #cv2.waitKey(1)
            
            imgarray = np.ndarray.flatten(state)
            nnOutput = net.activate(imgarray)
            
            state, rew, done, info = env.step(nnOutput)
          
            if info['x'] != current_x:
                current_x = info['x']
            else:
                counter += 1
                
            if info['level_end_bonus']:
                fitness_current += 1000000000
            
            if counter > 1200:
                done = True
                
            if done:
                novelty = 0
                if len(m.memory) and ((info['x'], info['y']) != start):
                    for x,y in m.memory.values():
                        novelty += (np.sqrt((x-info['x'])**2 + (y-info['y'])**2))/100.0
                else:
                    novelty = -1
                m.push((info['x'], info['y']))
                fitness_current += (novelty * (life / longest_life))
                print(genome_id, fitness_current, len(m.memory))
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
p.add_reporter(neat.Checkpointer(1, filename_prefix='checkpoints/neat-checkpoint-'))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
