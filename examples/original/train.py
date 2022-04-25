import igma.tasks.joust
import igma.tasks.legged_gym
from isaacgymenvs.train import launch_rlg_hydra
import hydra


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra_1(cfg):
    launch_rlg_hydra(cfg)


if __name__ == "__main__":
    launch_rlg_hydra_1()
