from pdb import set_trace as T

import nmmo
from nmmo.lib.material import Material
from nmmo.systems.skill import Skill

# Scripted models included with the baselines repository
from neural_mmo.baselines.scripted import baselines
import numpy as np

import neural_mmo.baselines.demos.minimal as minimal
from nmmo import Task
from ray import rllib
from collections import namedtuple
from nmmo.lib import material
from neural_mmo.ESE650.networks.policy import Simple
import os
from config import *

class Tier:
    REWARD_SCALE = 15
    EASY         = 4  / REWARD_SCALE
    NORMAL       = 6  / REWARD_SCALE
    HARD         = 11 / REWARD_SCALE


'''
DEfining tasks
'''
def exploration(realm, player):
    return player.history.exploration

def foraging(realm, player):
    return (player.skills.fishing.level + player.skills.hunting.level) / 2.0

Exploration = [
        Task(exploration, 32,  Tier.EASY),
        Task(exploration, 64,  Tier.NORMAL),
        Task(exploration, 127, Tier.HARD)]

Foraging = [
        Task(foraging, 20, Tier.EASY),
        Task(foraging, 35, Tier.NORMAL),
        Task(foraging, 50, Tier.HARD)]





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

        # Return signature includes fractal and material
        # Pass a zero array if fractal is not relevant
        return fractal, matl



class ForageEnv(nmmo.Env):
    def __init__(self,config):
        super(ForageEnv, self).__init__(config)
        self.init_water_capacity = None
        self.init_food_capacity = None

    def reset(self, idx=None, step=True):
        super(ForageEnv, self).reset(idx,step=False)
        self.SPAWN_DEPO_PLAYERS()
        self.num_steps = 0
        self.init_water_capacity = self.realm.map.depoTile.water_capacity
        self.init_food_capacity = self.realm.map.depoTile.food_capacity
        if step:
            self.obs, _, _, _ = self.step({})
        return self.obs

    def SPAWN_DEPO_PLAYERS(self):
        coords = np.where(np.array(self.realm.map.repr) == nmmo.Terrain.GRASS)
        materials = {mat.index: mat for mat in material.All}

        i = np.random.choice(np.arange(coords[0].shape[0]))
        self.realm.map.tiles[coords[0][i], coords[1][i]].reset(materials[nmmo.Terrain.DEPO],self.config)
        self.realm.map.depoTile = self.realm.map.tiles[coords[0][i], coords[1][i]]
        # from IPython import embed; embed()

        coords_depo = np.where(np.array(self.realm.map.repr)== nmmo.Terrain.DEPO)

        dx = np.abs(coords[0] - coords_depo[0])
        dy = np.abs(coords[1] - coords_depo[1])
        distance = dx+dy
        distance[distance<=self.config.MIN_SHARING_RANGE] = 2*self.config.TERRAIN_SIZE
        indices = np.argsort(distance)

        rows = coords[0][indices[:self.config.NENT]]
        cols = coords[1][indices[:self.config.NENT]]

        self.realm.spawned = True
        idx = 0
        for i in range(self.config.NENT):
            idx += 1
            r = rows[i]
            c = cols[i]
            assert not self.realm.map.tiles[r, c].occupied
            self.realm.players.spawnIndividual(r, c)


        #print("coords", coords[0][i], coords[1][i])


    def reward(self, player):
        # Default -1 reward for death
        # Infos returns per-task rewards
        reward, info = super().reward(player)

        # Inject new attribute
        curr_water_capacity = self.realm.map.depoTile.water_capacity
        curr_food_capacity = self.realm.map.depoTile.food_capacity

        # Team Reward added to each agent
        reward += (RESDEPOSIT.FOOD * (curr_food_capacity-self.init_food_capacity) +\
            RESDEPOSIT.WATER * (curr_water_capacity-self.init_water_capacity))

        return reward, info

    # Step in normal way
    # Sharing Resources amongst agents in the neighbourhood
    def step(self,actions):
        obs, rewards, dones, infos = super(ForageEnv, self).step(actions)
        self.num_steps+=1
        if self.config.RESOURCE_SHARING:
            group_dict = {}
            for entID, ent in self.realm.players.items():
                group_dict = {entID:entID}
            for entID, ent in self.realm.players.items():
                for entID2, ent2 in self.realm.players.items():
                    if group_dict[entID2] == group_dict[entID]:
                        continue
                    r2,c2 = self.realm.players.entities[group_dict[entID2]].pos
                    r,c = self.realm.players.entities[group_dict[entID]].pos
                    if np.abs(r2-r)+np.abs(c2-c)<=self.config.MIN_SHARING_RANGE:

                        group_dict[group_dict[entID2]] = group_dict[entID]
                        k = group_dict[entID]

                        # Finding ultimate parent
                        while group_dict[k]!=k:
                            k = group_dict[k]
                        group_dict[group_dict[entID2]] = k
            groups = {}
            for v in group_dict.values():
                groups[v] = []
                for k in group_dict.keys():
                    if group_dict[k]==v:
                        groups[v].append(k)

            for group in groups.values():
                food_tot = 0
                water_tot = 0
                for ID in group:
                    food_tot += self.realm.players.entities[ID].food
                    water_tot += self.realm.players.entities[ID].water
                food_tot/=len(group)
                water_tot/=len(group)
                for ID in group:
                    self.realm.players.entities[ID].food = food_tot
                    self.realm.players.entities[ID].water = water_tot
        return obs, rewards, dones, infos


def network_tester(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''

    # Environment accepts a config object
    env = env(config())
    env.reset()
    network = Simple(config)

    t = 0
    while True:
        # if render:
        #     env.render()

        # Scripted API computes actions
        obs, rewards, dones, infos = env.step(actions={})
        observations = {}
        for k in obs[1].keys():
            dict = {}
            for k1 in obs[1][k].keys():
                dict[k1] = []
            observations[k] = dict
        for agentid,object in obs.items():
            for item,val in obs[agentid].items():
                for k,v in obs[agentid][item].items():
                    observations[item][k].append(v)

        for k in observations.keys():
            for k1,v1 in observations[k].items():
                observations[k][k1] = np.array(v1)

        output = network(observations)
        # Later examples will use a fixed horizon
        t += 1
        if t >= horizon:
            break

    # Called at the end of simulation to obtain logs
    return env.terminal()



if __name__ == '__main__':
    #simulate(nmmo.Env, Config, render=True)
    network_tester(ForageEnv,ForageConfigDebug,render=True)