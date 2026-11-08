"""Evaluate MCTS with parameters from config/mcts.yaml."""

from game2048.runner import run_method


def main(argv=None):
    return run_method("mcts", argv)


if __name__ == "__main__":
    main()
