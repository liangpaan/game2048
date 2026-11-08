"""Evaluate Expectimax with parameters from config/expectimax.yaml."""

from game2048.runner import run_method


def main(argv=None):
    return run_method("expectimax", argv)


if __name__ == "__main__":
    main()
