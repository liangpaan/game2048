"""Evaluate ppo with parameters from config/ppo.yaml."""

from game2048.runner import run_method


def main(argv=None):
    return run_method("ppo", argv)


if __name__ == "__main__":
    main()
