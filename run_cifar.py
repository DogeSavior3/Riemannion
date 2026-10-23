import hydra
from omegaconf import DictConfig

from src.experiments.cifar import run_cifar


@hydra.main(version_base=None, config_path="src/configs", config_name="cifar")
def main(cfg: DictConfig):
    run_cifar(cfg)


if __name__ == "__main__":
    main()
