from gym.envs.registration import register

register(
    id='BipedalWalkerStump1-v0',
    entry_point='custom_env.bipedal_walker_stump1:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerStump2-v0',
    entry_point='custom_env.bipedal_walker_stump2:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerPit1-v0',
    entry_point='custom_env.bipedal_walker_pit1:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerPit2-v0',
    entry_point='custom_env.bipedal_walker_pit2:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerStairs1-v0',
    entry_point='custom_env.bipedal_walker_stairs1:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerHardcoreStump1-v0',
    entry_point='custom_env.bipedal_walker_hardcore_stump1:BipedalWalkerHardcoreEdit2'
)
