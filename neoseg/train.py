import lightning
from lightning.pytorch.cli import LightningCLI


def main():
    cli = LightningCLI(
        trainer_class=lightning.pytorch.Trainer,
        seed_everything_default=0
    )
    return cli


if __name__ == "__main__":
    main()
