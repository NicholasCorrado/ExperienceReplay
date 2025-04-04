MEMDISK = {
    # 'CartPole-v1': (2.1, 2),
    # 'LunarLander-v2': (2.1, 2),
    # 'Discrete2D100-v0': (2.1, 2),

    'InvertedPendulum-v4': (2.1, 2),
    'InvertedDoublePendulum-v4': (2.1, 2),
    'Swimmer-v4': (2.1, 2),
    'HalfCheetah-v4': (2.1, 2),
    'Hopper-v4': (2.1, 2),
    'Walker2d-v4': (2.1, 2),
    'Ant-v4': (2.1, 2),
    'Humanoid-v4': (3, 3),
}

TIMESTEPS = {
    # 'CartPole-v1': int(0.5e6),
    # 'LunarLander-v2': int(0.5e6),
    # 'Discrete2D100-v0': int(0.5e6),

    'InvertedPendulum-v4': int(300e3),
    'InvertedDoublePendulum-v4': int(300e3),
    'Swimmer-v4': int(1e6),
    'HalfCheetah-v4': int(1e6),
    'Hopper-v4': int(1e6),
    'Walker2d-v4': int(1e6),
    'Ant-v4': int(1e6),
    'Humanoid-v4': int(1e6),
}

PARAMS = {
    'CartPole-v1': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'LunarLander-v2': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Discrete2D100-v0': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Swimmer-v4': {
        'lr': 1e-3,
        'ns': 8192,
    },
    'HalfCheetah-v4': {
        'lr': 1e-4,
        'ns': 4096,
    },
    'Hopper-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Walker2d-v4': {
        'lr': 1e-4,
        'ns': 2048,
    },
    'Ant-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Humanoid-v4': {
        'lr': 1e-4,
        'ns': 8192,
    },
}


PARAMS = {
    'CartPole-v1': {
        'lr': 1e-4,
        'ns': 256,
    },
    'LunarLander-v2': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Discrete2D100-v0': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Swimmer-v4': {
        'lr': 1e-3,
        'ns': 8192,
    },
    'HalfCheetah-v4': {
        'lr': 1e-3,
        'ns': 1024,
    },
    'Hopper-v4': {
        'lr': 1e-3,
        'ns': 4096,
    },
    'Walker2d-v4': {
        'lr': 1e-4,
        'ns': 2048,
    },
    'Ant-v4': {
        'lr': 1e-4,
        'ns': 1024,
    },
    'Humanoid-v4': {
        'lr': 1e-4,
        'ns': 8192,
    },
}