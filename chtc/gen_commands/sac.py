import os

from chtc.gen_commands.hyperparameters import PARAMS, TIMESTEPS, MEMDISK

if __name__ == "__main__":

    os.makedirs('../commands', exist_ok=True)
    f = open(f"../commands/example.txt", "w")

    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']
    env_ids = [
        # 'InvertedPendulum-v4',
        # 'InvertedDoublePendulum-v4',
        'Swimmer-v4','Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4'
    ]

    b = 1
    for env_id in env_ids:
        # for lr in [1e-4, 3e-4, 1e-3]:
        for plr, qlr in [(3e-4, 1e-3)]:
            for bs in [256]:
                command = (
                    f"python sac_continuous_action.py --output_subdir plr_{plr}/qlr_{qlr}/bs_{bs}"
                    f" --env_id {env_id}"
                    f" --policy_lr {plr}"
                    f" --q_lr {qlr}"
                    f" --batch_size {bs}"
                    f" --total_timesteps {TIMESTEPS[env_id]}"
                    f" --eval_episodes 20"
                )


                mem, disk = MEMDISK[env_id]
                command = f"{mem},{disk},{command}"
                print(command)
                f.write(command + "\n")


    for env_id in env_ids:
        # for lr in [1e-4, 3e-4, 1e-3]:
        for plr, qlr in [(3e-4, 1e-3)]:
            for bs in [256]:
                for t in [1, 0.5, 0.25, 0.125]:

                    command = (
                        f"python sac_continuous_action.py --output_subdir plr_{plr}/qlr_{qlr}/bs_{bs}"
                        f" --env_id {env_id}"
                        f" --policy_lr {plr}"
                        f" --q_lr {qlr}"
                        f" --batch_size {bs}"
                        f" --total_timesteps {TIMESTEPS[env_id]}"
                        f" --eval_episodes 20"
                        f" --adaptive "
                        f" --temperature {t}"
                    )

                    mem, disk = MEMDISK[env_id]
                    command = f"{mem},{disk},{command}"
                    print(command)
                    f.write(command + "\n")
