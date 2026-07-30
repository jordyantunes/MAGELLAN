from environment import VectorizedEnv, generate_goals, list_to_dict
from utils.tests import test_policy, test_lp
from lamorel import Caller, lamorel_init
import numpy as np
from little_zoo import LittleZoo

agent = None#Caller(config=)
nb_test_envs = 256
seed = 0
goals_distribution = [20000, 4000, 800, 160, 32]

test_envs = VectorizedEnv(nb_test_envs, False, seed, True)
test_goals = generate_goals(test_envs.envs[0], seed, goals_distribution, False)

all_grasp = [g for g in test_goals['grasp'] if g in test_goals['goals'].keys()]
all_grow_plants = [g for g in test_goals['grow_plants'] if g in test_goals['goals'].keys()]
all_grow_herbivores = [g for g in test_goals['grow_herbivores'] if g in test_goals['goals'].keys()]
all_grow_carnivores = [g for g in test_goals['grow_carnivores'] if g in test_goals['goals'].keys()]
# grasp = list(np.random.choice(all_grasp, 2048, replace=True))
# grow_plants = list(np.random.choice(all_grow_plants, 2048, replace=True))
# grow_herbivores = list(np.random.choice(all_grow_herbivores, 2048, replace=True))
# grow_carnivores = list(np.random.choice(all_grow_carnivores, 2048, replace=True))
grasp = list(np.random.choice(all_grasp, 64, replace=True))
grow_plants = list(np.random.choice(all_grow_plants, 64, replace=True))
grow_herbivores = list(np.random.choice(all_grow_herbivores, 64, replace=True))
grow_carnivores = list(np.random.choice(all_grow_carnivores, 64, replace=True))
goals = grasp + grow_plants + grow_herbivores + grow_carnivores

print(goals[0])
observations, infos = test_envs.reset([test_goals['goals'][g] for g in goals])
# print(observations[0])
print(infos)
# print(infos)
# for obj in [test_goals['goals'][g] for g in goals][1:]:
#     print(obj)

# result = test_policy(test_envs, test_goals, agent)