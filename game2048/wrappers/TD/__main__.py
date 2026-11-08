"""Evaluate TD with parameters from config/td.yaml."""

from game2048.runner import run_method


def main(argv=None):
    return run_method("td", argv)


if __name__ == "__main__":
    main()
