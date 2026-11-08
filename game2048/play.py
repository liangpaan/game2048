"""Show one demo game for MCTS, Expectimax, TD and PPO."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time

from game2048.runner import (
    CONFIG_DIR, METHODS, create_agent, episode_result, load_config, merge_config,
    read_yaml, resolve_path, save_results, summarize,
)
from game2048.wrappers.env import ExperimentEnv


# The windowed entry point is intentionally a visual demo. Batch evaluation
# belongs to game2048.runner and the per-method wrapper entry points.
DEMO_EPISODES = 1


def _agent_worker(method, config, requests, responses):
    """Keep model loading and search off the window's event thread."""
    try:
        agent = create_agent(method, config)
        responses.put(("ready", None))
        while True:
            task = requests.get()
            if task is None:
                return
            board, seed = task
            if seed is not None and hasattr(agent, "rng"):
                agent.rng.seed(seed + 100_000)
            responses.put(("action", agent.choose(board)))
    except Exception as exc:
        responses.put(("error", str(exc)))


def _draw(pygame, screen, fonts, method, info, grid, steps, results, total, status):
    width, height = screen.get_size()
    ink, muted = (59, 64, 73), (110, 116, 125)
    screen.fill((247, 248, 250))

    def label(text, position, font="small", color=ink):
        screen.blit(fonts[font].render(text, True, color), position)

    label(METHODS[method], (20, 16), "title")
    episode = min(len(results) + bool(info["legal_actions"]), total)
    label(f"Game {episode} / {total}    Steps {steps}", (20, 50), color=muted)
    for x, caption, value in ((20, "SUM", info["board_sum"]),
                              (width // 2 + 4, "MAX TILE", info["max_tile"])):
        pygame.draw.rect(screen, (231, 234, 239), (x, 78, width // 2 - 28, 48), border_radius=8)
        label(caption, (x + 10, 83), "caption", muted)
        label(str(value), (x + 10, 98), "number")

    side = min(width - 40, height - 220)
    left, top, gap = (width - side) // 2, 140, 8
    cell = (side - gap * 4) // 3
    pygame.draw.rect(screen, (177, 168, 156), (left, top, side, side), border_radius=10)
    colors = {0: (204, 195, 183), 2: (238, 230, 219), 4: (236, 224, 197),
              8: (242, 179, 119), 16: (245, 151, 96), 32: (246, 125, 94),
              64: (244, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
              512: (237, 199, 79), 1024: (237, 195, 62)}
    for row in range(3):
        for col in range(3):
            value = int(grid[row, col])
            rect = pygame.Rect(left + gap + col * (cell + gap),
                               top + gap + row * (cell + gap), cell, cell)
            pygame.draw.rect(screen, colors.get(value, (65, 64, 62)), rect, border_radius=6)
            if value:
                font = fonts["tile"] if value < 1000 else fonts["tile_small"]
                text = font.render(str(value), True, ink if value <= 4 else (255, 251, 240))
                screen.blit(text, text.get_rect(center=rect.center))
    summary = summarize(results)
    mean = "--" if summary["mean_sum"] is None else f"{summary['mean_sum']:.2f}"
    label(f"Mean sum  {mean}", (20, height - 62))
    label(f"Passed  {summary['passed']} / {len(results)}", (width // 2 + 4, height - 62))
    label(status, (20, height - 30), "caption", muted)


def run_window(method, config, display, index, stop):
    """One process owns one Pygame window and its own game/RNG/model."""
    import pygame

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = ExperimentEnv()
    requests, responses = queue.Queue(), queue.Queue()
    results = []
    error = None
    worker = None
    try:
        pygame.display.init()
        pygame.font.init()
        screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
        width, height, gap = display["width"], display["height"], display["gap"]
        left = max(0, (screen_width - width * 2 - gap) // 2)
        top = max(0, (screen_height - height * 2 - gap - 60) // 2)
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{left + index % 2 * (width + gap)},{top + index // 2 * (height + gap + 30)}"
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"2048 | {METHODS[method]}")
        fonts = {name: pygame.font.SysFont("dejavusans", size, bold=bold) for name, size, bold in (
            ("title", 26, True), ("small", 16, False), ("caption", 13, False),
            ("number", 23, True), ("tile", 42, True), ("tile_small", 32, True),
        )}
        settings = config["evaluation"]
        total, first_seed = DEMO_EPISODES, settings["seed"]
        observation, info = env.reset(seed=first_seed)
        steps, ready, busy, finished = 0, False, False, False
        next_move = 0.0
        status = "Loading agent..."
        worker = threading.Thread(target=_agent_worker,
                                  args=(method, config, requests, responses))
        worker.start()
        clock = pygame.time.Clock()
        print(json.dumps({"event": "window_opened", "method": method}), flush=True)
        while not stop.is_set():
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                break
            now = time.monotonic()
            try:
                kind, value = responses.get_nowait()
            except queue.Empty:
                kind = None
            if kind == "ready":
                ready = True
                status = "Running - goal: 256"
                print(json.dumps({"event": "agent_ready", "method": method}), flush=True)
            elif kind == "error":
                error = value
                print(json.dumps({"event": "agent_error", "method": method, "error": error}), flush=True)
                stop.set()
                break
            elif kind == "action":
                busy = False
                if value not in info["legal_actions"]:
                    raise ValueError(f"{method} selected illegal action {value}")
                observation, _, terminated, truncated, info = env.step(value)
                if truncated:
                    raise RuntimeError("player games must finish naturally")
                steps += 1
                next_move = now + 1.0 / display["fps"]
                status = "256 reached - continuing" if info["target_reached"] else "Running - goal: 256"
                if terminated:
                    result = {"episode": len(results) + 1,
                              **episode_result(first_seed + len(results), info, steps)}
                    results.append(result)
                    print(json.dumps({"event": "episode_end", "method": method, **result}), flush=True)
                    finished = len(results) == total
                    next_move = now + display["episode_pause"]
                    status = "Finished - close window to exit" if finished else "Game over - next game shortly"
                    if finished:
                        save_results(results, settings["output"])
                        print(json.dumps({"event": "evaluation_end", "method": method,
                                          **summarize(results)}), flush=True)
            if ready and not busy and not error and now >= next_move and not finished:
                if results and not info["legal_actions"]:
                    observation, info = env.reset(seed=first_seed + len(results))
                    steps = 0
                board = tuple(int(tile) for tile in observation.flat)
                requests.put((board, first_seed + len(results) if steps == 0 else None))
                busy = True
            _draw(pygame, screen, fonts, method, info, observation, steps, results, total, status)
            pygame.display.flip()
            clock.tick(60)
    finally:
        requests.put(None)
        env.close()
        pygame.quit()
        if worker is not None:
            # Torch/native solver objects must be released before Python exits.
            worker.join()
    if error:
        raise SystemExit(1)


def load_player_config(path=None):
    config = read_yaml(CONFIG_DIR / "player.yaml")
    if path is not None:
        config = merge_config(config, read_yaml(path))
    for name, minimum in (("width", 320), ("height", 480), ("gap", 0)):
        if type(config[name]) is not int or config[name] < minimum:
            raise ValueError(f"player {name} must be an integer >= {minimum}")
    if not isinstance(config["fps"], (int, float)) or config["fps"] <= 0:
        raise ValueError("player fps must be positive")
    if not isinstance(config["episode_pause"], (int, float)) or config["episode_pause"] < 0:
        raise ValueError("player episode_pause must be nonnegative")
    if type(config["close_when_done"]) is not bool:
        raise ValueError("player close_when_done must be true or false")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="override config/player.yaml")
    args = parser.parse_args(argv)
    try:
        display = load_player_config(args.config)
        configs = {method: load_config(method, resolve_path(path))
                   for method, path in display["methods"].items()}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    context = mp.get_context("spawn")
    stop = context.Event()
    processes = [context.Process(target=run_window, args=(method, configs[method], display, index, stop),
                                 name=f"2048-{method}")
                 for index, method in enumerate(METHODS)]
    previous = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                stop.set()
            for process in processes:
                process.join(timeout=0.05)
    finally:
        stop.set()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(any(process.exitcode not in (None, 0) for process in processes))


if __name__ == "__main__":
    raise SystemExit(main())
