from sgcr.validate import validate
import os
from yaecs import Experiment
from sgcr.train import train
from config.config import CVConfig

def main(config, tracker):
    """ Main entry point. """
    if config.run_mode == "train":
        train(config, tracker)
    elif config.run_mode in ("val", "valid", "validate", "validation"):
        validate(config, tracker, mode="val")
    elif config.run_mode == "test":
        validate(config, tracker, mode="test")
    else:
        raise ValueError(f"Unknown mode: {config.run_mode}")


if __name__ == "__main__":
    fallback = os.path.join(os.path.dirname(__file__), "config", "experiments", "debug.yaml")
    configuration = CVConfig.build_from_argv(fallback=fallback, overwriting_regime="locked")
    Experiment(configuration, main).run(run_description=configuration.experiment_purpose)
 