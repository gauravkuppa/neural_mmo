'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo
from nmmo.lib.material import Material
from nmmo.systems.skill import Skill
# Scripted models included with the baselines repository
from scripted import baselines


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


class Depo(Material):
   tex = 'depo'
   index = 6

   def __init__(self, config):
      if config.game_system_enabled('Resource'):
          pass

class DepoSkill(Skill):
   def __init__(self, skillGroup):
      super().__init__(skillGroup)
      self.setExpByLevel(self.config.BASE_HEALTH)
   
   def update(self, realm, entity):
      health = entity.resources.health
      food   = entity.resources.food
      water  = entity.resources.water
      config = self.config

      if not config.game_system_enabled('Resource'):
         return

      # how to get Tile object?
      depo.capacity += food//2
      depo.capacity += water//2

      food -= food//2
      water -= water//2
    
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



if __name__ == '__main__':
    simulate(nmmo.Env, Config, render=True, horizon=10)
