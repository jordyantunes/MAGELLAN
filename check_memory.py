import tracemalloc
import sys
sys.path.insert(0, 'magellan')

from little_zoo import LittleZoo
from environment import generate_goals

tracemalloc.start()

env = LittleZoo(train=True, seed=0)
snapshot_before = tracemalloc.take_snapshot()

goals_distribution = [20000, 4000, 800, 160, 32]
goals = generate_goals(env, seed=0, distribution=goals_distribution, filter_test=False)

snapshot_after = tracemalloc.take_snapshot()

stats = snapshot_after.compare_to(snapshot_before, 'lineno')
print(f"Total goals generated: {len(goals['goals'])}")
print(f"  impossibles: {len(goals['impossibles'])}, grasp: {len(goals['grasp'])}, "
      f"grow_plants: {len(goals['grow_plants'])}, grow_herbivores: {len(goals['grow_herbivores'])}, "
      f"grow_carnivores: {len(goals['grow_carnivores'])}")
print(f"\nTop 10 memory consumers during generate_goals:")
for stat in stats[:10]:
    print(stat)

current, peak = tracemalloc.get_traced_memory()
print(f"\nPeak memory during generate_goals: {peak / 1e6:.1f} MB")
