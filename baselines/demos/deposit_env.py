'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo
from nmmo.lib.material import Material
from nmmo.systems.skill import Skill
# Scripted models included with the baselines repository
from scripted import baselines
import numpy as np
from demos import minimal

def simulate(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''

    # Environment accepts a config object
    env = env(config())
    env.reset()

    t = 0
    while True:
        if render:
            env.render()

        # Scripted API computes actions
        obs, rewards, dones, infos = env.step(actions={})

        # Later examples will use a fixed horizon
        t += 1
        if t >= horizon:
            break

    # Called at the end of simulation to obtain logs
    return env.terminal()






class CustomMapGenerator(nmmo.MapGenerator):
    '''Subclass the base NMMO Map Generator'''
    def generate_map(self, idx):
        '''Override the default per-map generation method'''
        config  = self.config
        size    = config.TERRAIN_SIZE

        # Create fractal and material placeholders
        fractal = np.zeros((size, size)) #Unused in demo
        matl    = np.zeros((size, size), dtype=object)

        for r in range(size):
            for c in range(size):
                linf = max(abs(r - size//2), abs(c - size // 2))

                # Set per-tile materials
                if linf < 4:
                    matl[r, c] = nmmo.Terrain.STONE
                elif linf < 8:
                    matl[r, c] = nmmo.Terrain.WATER
                elif linf < 12:
                    matl[r, c] = nmmo.Terrain.FOREST
                elif linf <= size//2 - config.TERRAIN_BORDER:
                    matl[r, c] = nmmo.Terrain.GRASS
                else:
                    matl[r, c] = nmmo.Terrain.LAVA
        coords = np.where(matl == nmmo.Terrain.GRASS)
        #for _ in range(50): 
        i = np.random.choice(np.arange(coords[0].shape[0]))
        matl[coords[0][i], coords[1][i]] = nmmo.Terrain.DEPO
        #from IPython import embed; embed()
        print("coords", coords[0][i], coords[1][i])
        # Return signature includes fractal and material
        # Pass a zero array if fractal is not relevant
        return fractal, matl

class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    AGENTS    = [baselines.Meander, baselines.Forage]

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True
    MAP_GENERATOR = CustomMapGenerator
    GENERATE_MAP_PREVIEWS = True
    #NENT = 100



if __name__ == '__main__':
    simulate(nmmo.Env, Config, render=True)
