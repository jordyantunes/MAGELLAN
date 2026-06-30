'''

'''

import numpy as np
from collections import defaultdict, OrderedDict
from little_zoo import LittleZoo

class VectorizedEnv():
    
    def __init__(self, num_envs, train, seed=None, colors=False):
        self.envs = [LittleZoo(colors = colors, train = train, seed = seed + i * 100) for i in range(num_envs)]
    
    def reset(self, goals=None):
        if goals is not None:
            results = [env.reset(goal) for env, goal in zip(self.envs, goals)]
        else:
            results = [env.reset() for env in self.envs]
        observations, infos = zip(*results)
        return list(observations), list_to_dict(infos)
    
    def reset_at(self, index, goal):
        observation, infos = self.envs[index].reset(goal)
        return observation, infos
    
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, _, infos = zip(*results)
        return list(observations), rewards, dones, False, list_to_dict(infos)
    

def generate_goals(env, seed, distribution, filter_test):

    rng = np.random.default_rng(seed)

    furnitures = env.env_params['categories']['furniture'][:6]
    plants = env.env_params['categories']['plant'][:6]
    herbivores = env.env_params['categories']['herbivore'][:6]
    carnivores = env.env_params['categories']['carnivore'][:6]
    supplies = env.env_params['categories']['supply']

    furnitures_test = env.env_params['categories']['furniture'][6:]
    plants_test = env.env_params['categories']['plant'][6:]
    herbivores_test = env.env_params['categories']['herbivore'][6:]
    carnivores_test = env.env_params['categories']['carnivore'][6:]

    objects = furnitures + plants + herbivores + carnivores + supplies
    objects_test = furnitures_test + plants_test + herbivores_test + carnivores_test

    plants_set = set(plants)
    herbivores_set = set(herbivores)
    carnivores_set = set(carnivores)
    furnitures_set = set(furnitures)
    supplies_set = set(supplies)
    plants_test_set = set(plants_test)
    herbivores_test_set = set(herbivores_test)
    carnivores_test_set = set(carnivores_test)
    furnitures_test_set = set(furnitures_test)

    colors = env.env_params['attributes']['colors'] if 'colors' in env.env_params['admissible_attributes'] else None
        
    def get_name(obj):
        if colors:
            color = rng.choice(colors)

            if obj in plants_set | plants_test_set:
                return f'{color} {obj} seed'
            elif obj in herbivores_set | carnivores_set | herbivores_test_set | carnivores_test_set:
                return f'{color} baby {obj}'
            else:
                return f'{color} {obj}'
        else:
            if obj in plants_set | plants_test_set:
                return obj + ' seed'
            elif obj in herbivores_set | carnivores_set | herbivores_test_set | carnivores_test_set:
                return 'baby ' + obj
            else:
                return obj

    def _reservoir_add(reservoir, item, count, k):
        if count < k:
            reservoir.append(item)
        else:
            j = int(rng.integers(0, count + 1))
            if j < k:
                reservoir[j] = item

    goals = OrderedDict()
    impossibles, grasp, grow_plants, grow_herbivores, grow_carnivores = [], [], [], [], []

    if env.train:

        caps = distribution  # [n_impossibles, n_grasp, n_grow_plants, n_grow_herbivores, n_grow_carnivores]
        reservoirs = [[], [], [], [], []]
        counts = [0, 0, 0, 0, 0]

        for e1 in objects:
            e1_name = get_name(e1)
            for e2 in objects:
                e2_name = get_name(e2)
                for e3 in objects:
                    e3_name = get_name(e3)
                    for e4 in objects:
                        e4_name = get_name(e4)
                        if colors:
                            seen = {
                                (e1, e1_name),
                                (e2, e2_name),
                                (e3, e3_name),
                                (e4, e4_name),
                            }
                            seen_types = {x[0] for x in seen}
                        else:
                            seen = {e1, e2, e3, e4}
                            seen_types = seen

                        has_water = 'water' in seen_types
                        has_plant = bool(seen_types & plants_set)
                        has_herbivore = bool(seen_types & herbivores_set)
                        for o in objects:
                            if colors:
                                for color in colors:
                                    g = (f'Goal: {{t}} {color} {o}\n'
                                        f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                                        'You are standing on: nothing\n'
                                        'Inventory (0/2): empty\n'
                                        'Action: ')
                                    meta = (o, e1, e2, e3, e4, color)
                                    o_in_scene = any(
                                        obj == o and color in name
                                        for obj, name in seen
                                    )
                            else:
                                g = (f'Goal: {{t}} {o}\n'
                                    f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                                    'You are standing on: nothing\n'
                                    'Inventory (0/2): empty\n'
                                    'Action: ')
                                meta = (o, e1, e2, e3, e4)
                                o_in_scene = o in seen

                            for t in ('Grasp', 'Grow'):
                                goal = g.replace('{t}', t)
                                if colors:
                                    full_meta = (t + ' ' + color + ' ' + o,) + meta[1:]
                                else:
                                    full_meta = (t + ' ' + o,) + meta[1:]

                                if (not o_in_scene
                                        or t == 'Grow' and (
                                            o in furnitures_set or o in supplies_set
                                            or not has_water
                                            or o in herbivores_set | carnivores_set and not has_plant
                                            or o in carnivores_set and not has_herbivore)):
                                    cat = 0
                                elif t == 'Grasp':
                                    cat = 1
                                elif o in plants_set:
                                    cat = 2
                                elif o in herbivores_set:
                                    cat = 3
                                elif o in carnivores_set:
                                    cat = 4
                                else:
                                    raise ValueError(f'Invalid goal: {goal}')

                                _reservoir_add(reservoirs[cat], (goal, full_meta), counts[cat], caps[cat])
                                counts[cat] += 1

        cat_lists = [impossibles, grasp, grow_plants, grow_herbivores, grow_carnivores]
        for reservoir, cat_list in zip(reservoirs, cat_lists):
            for goal, meta in reservoir:
                goals[goal] = meta
                cat_list.append(goal)

    else:

        all_goals = OrderedDict()

        for e2 in objects:
            e2_name = get_name(e2)
            for e3 in objects:
                e3_name = get_name(e3)
                for e4 in objects:
                    e4_name = get_name(e4)
                    for o in objects_test:
                        e1 = o
                        e1_name = get_name(e1)
                        seen = {e1, e2, e3, e4}
                        has_water = 'water' in seen
                        has_plant = bool(seen & plants_set)
                        has_herbivore = bool(seen & herbivores_set)
                        for t in ('Grasp', 'Grow'):
                            if colors:
                                for color in colors:
                                    g = (f'Goal: {t} {color} {o}\n'
                                        f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                                        'You are standing on: nothing\n'
                                        'Inventory (0/2): empty\n'
                                        'Action: ')
                            else:
                                g = (f'Goal: {t} {o}\n'
                                    f'You see: {e1_name}, {e2_name}, {e3_name}, {e4_name}\n'
                                    'You are standing on: nothing\n'
                                    'Inventory (0/2): empty\n'
                                    'Action: ')

                            if (o not in seen
                                    or t == 'Grow' and (
                                        o in furnitures_test_set or o in supplies_set
                                        or not has_water
                                        or o in herbivores_test_set | carnivores_test_set and not has_plant
                                        or o in carnivores_test_set and not has_herbivore)):
                                impossibles.append(g)
                            elif t == 'Grasp':
                                grasp.append(g)
                            elif o in plants_test_set:
                                grow_plants.append(g)
                            elif o in herbivores_test_set:
                                grow_herbivores.append(g)
                            elif o in carnivores_test_set:
                                grow_carnivores.append(g)
                            else:
                                raise ValueError('Invalid object')

                            if colors:
                                all_goals[g] = (t + ' ' + color + ' ' + o, e1, e2, e3, e4)
                            else:
                                all_goals[g] = (t + ' ' + o, e1, e2, e3, e4)

        if filter_test:
            def _sample(lst, k):
                if len(lst) <= k:
                    return lst
                idx = rng.choice(len(lst), k, replace=False)
                return [lst[i] for i in idx]

            for lst in [
                _sample(impossibles, distribution[0]),
                _sample(grasp, distribution[1]),
                _sample(grow_plants, distribution[2]),
                _sample(grow_herbivores, distribution[3]),
                _sample(grow_carnivores, distribution[4]),
            ]:
                for g in lst:
                    goals[g] = all_goals[g]

            impossibles = _sample(impossibles, distribution[0])
            grasp = _sample(grasp, distribution[1])
            grow_plants = _sample(grow_plants, distribution[2])
            grow_herbivores = _sample(grow_herbivores, distribution[3])
            grow_carnivores = _sample(grow_carnivores, distribution[4])
        else:
            goals = all_goals

    return {'goals': goals, 'impossibles': impossibles, 'grasp': grasp, 'grow_plants': grow_plants,
            'grow_herbivores': grow_herbivores, 'grow_carnivores': grow_carnivores}
    
    
def list_to_dict(list_):
    dict_ = defaultdict(list)
    for d in list_:
        for key, value in d.items():
            dict_[key].append(value)
    return dict(dict_)