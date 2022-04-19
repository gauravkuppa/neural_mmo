import nmmo
import os
from neural_mmo.baselines.scripted import baselines
class ForagingGameSystems(nmmo.config.Resource, nmmo.config.Deposit): pass

class Debug:
   '''Scale arguments for debugging

   Requires a 32 core 1 GPU machine
   '''

   NUM_GPUS                = 0
   NUM_WORKERS             = 2
   EVALUATION_NUM_WORKERS  = 1
   EVALUATION_NUM_EPISODES = 1

class Train:
   '''Scale arguments for debugging

   Requires a 32 core 1 GPU machine
   '''

   NUM_GPUS                = 1
   NUM_WORKERS             = 36
   EVALUATION_NUM_WORKERS  = 2
   EVALUATION_NUM_EPISODES = 2

class RLlib:
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters

   Configure NUM_GPUS and NUM_WORKERS for your hardware
   Note that EVALUATION_NUM_WORKERS cores are reserved for evaluation
   and one additional core is reserved for the driver process.
   Therefore set NUM_WORKERS <= cores - EVALUATION_NUM_WORKERS - 1
   '''

   #Run in train/evaluation mode
   EVALUATE     = False
   N_TRAIN_MAPS = 256

   @property
   def MODEL(self):
      return self.__class__.__name__

   @property
   def PATH_MAPS(self):
      maps = super().PATH_MAPS
      if self.EVALUATE:
          self.TERRAIN_FLIP_SEED = True
          return os.path.join(maps, 'evaluation')
      return os.path.join(maps, 'training')

   @property
   def NMAPS(self):
      if not self.EVALUATE:
          return self.N_TRAIN_MAPS
      return super().NMAPS

   @property
   def TRAIN_BATCH_SIZE(self):
      return 64 * 256 * self.NUM_WORKERS

   #Checkpointing. Resume will load the latest trial, e.g. to continue training
   #Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
   EXPERIMENT_DIR          = 'experiments'
   RESUME                  = False

   RESTORE                 = False
   RESTORE_ID              = 'Baseline' #Experiment name suffix
   RESTORE_CHECKPOINT      = 1000

   #Policy specification
   #EVAL_AGENTS             = [baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS                  = [nmmo.Agent]
   TASKS                   = []

   #Hardware and debug
   NUM_GPUS_PER_WORKER     = 0
   LOCAL_MODE              = False
   LOG_LEVEL               = 1

   #Training and evaluation settings
   EVALUATION_INTERVAL     = 1
   EVALUATION_PARALLEL     = True
   TRAINING_ITERATIONS     = 1000
   KEEP_CHECKPOINTS_NUM    = 3
   CHECKPOINT_FREQ         = 1
   LSTM_BPTT_HORIZON       = 16
   NUM_SGD_ITER            = 1

   #Model
   SCRIPTED                = None
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64
   DEVICE                  = 'cuda'

   #Reward
   COOPERATIVE             = False
   TEAM_SPIRIT             = 0.0

   #SGD
   SGD_MINIBATCH_SIZE = 128

class RESDEPOSIT:
    FOOD = 10
    WATER = 10


class Small(RLlib, nmmo.config.Small):
    '''Small scale Neural MMO training setting

    Features up to 64 concurrent agents and 32 concurrent NPCs,
    64 x 64 maps (excluding the border), and 128 timestep horizons'''

    # Memory/Batch Scale
    ROLLOUT_FRAGMENT_LENGTH = 128
    SGD_MINIBATCH_SIZE = 128

    # Horizon
    TRAIN_HORIZON = 128
    EVALUATION_HORIZON = 128


class ForageConfigDebug(Small, ForagingGameSystems,RESDEPOSIT,Debug):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    AGENTS    = [nmmo.Agent]
    EVAL_AGENTS = 4*[baselines.Meander,baselines.Forage]

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True
    #MAP_GENERATOR = CustomMapGenerator
    GENERATE_MAP_PREVIEWS = True
    NMAPS = 10
    NENT = 8

    #NENT = 100
    # Sharing
    MIN_SHARING_RANGE = 2
    RESOURCE_SHARING = False

    # Map Generation
    TERRAIN_LAVA = 0.0
    TERRAIN_WATER = 0.35
    TERRAIN_GRASS = 0.55
    TERRAIN_FOREST = 0.90
    TERRAIN_BORDER = 10


    # Override Spawn
    def SPAWN_NOTHING(self):
        return None,None
    @property
    def SPAWN(self):
        return self.SPAWN_NOTHING

class ForageConfigTrain(nmmo.config.Small, ForagingGameSystems,RLlib,RESDEPOSIT,Train):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    AGENTS    = [nmmo.Agent]

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True
    #MAP_GENERATOR = CustomMapGenerator
    GENERATE_MAP_PREVIEWS = True
    NMAPS = 10
    NENT = 8

    #NENT = 100
    # Sharing
    MIN_SHARING_RANGE = 2
    RESOURCE_SHARING = False

    # Map Generation
    TERRAIN_LAVA = 0.0
    TERRAIN_WATER = 0.35
    TERRAIN_GRASS = 0.55
    TERRAIN_FOREST = 0.90
    TERRAIN_BORDER = 10

    # Override Spawn
    def SPAWN_NOTHING(self):
        return None,None
    @property
    def SPAWN(self):
        return self.SPAWN_NOTHING
