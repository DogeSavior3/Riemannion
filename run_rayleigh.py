import hydra
from omegaconf import DictConfig

from src.experiments.rayleigh import run_rayleigh


@hydra.main(version_base=None, config_path="src/configs", config_name="rayleigh")
def main(cfg: DictConfig):
    run_rayleigh(cfg)


if __name__ == "__main__":
    main()
