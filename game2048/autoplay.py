"""Run one of the three retained autonomous 3x3 agents."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .selected_env import Game2048Env
from .agents import Agent


def episode(agent, seed, record=False):
    agent.reset(seed)
    env=Game2048Env()
    grid,info=env.reset(seed=seed)
    frames=[dict(board=grid.tolist(),action=None,**{k:v for k,v in info.items() if k!="action_mask"})]
    start=time.perf_counter()
    decisions=0.0
    total_reward=0.0
    for _ in range(4096):
        tick=time.perf_counter()
        action=agent.act(grid)
        decisions+=time.perf_counter()-tick
        if action<0:
            break
        if not info["action_mask"][action]:
            raise RuntimeError("Agent and Gym disagree on legal actions")
        grid,reward,done,truncated,info=env.step(action)
        total_reward+=reward
        if record:
            frames.append(dict(board=grid.tolist(),action=action,reward=reward,
                              **{k:v for k,v in info.items() if k!="action_mask"}))
        if done or truncated:
            break
    env.close()
    result=dict(seed=seed,tile_sum=int(grid.sum()),max_tile=int(grid.max()),
                steps=info["steps"],reached_target=bool(grid.max()>=256),
                total_reward=total_reward,seconds=time.perf_counter()-start,
                decision_seconds=decisions,truncated=info["steps"]>=4096)
    return result,frames


def interactive(agent,seed,fps,screenshot=None):
    import pygame
    pygame.init()
    screen=pygame.display.set_mode((720,850))
    policy_name=agent.name
    pygame.display.set_caption("3 x 3 2048 | "+policy_name)
    clock=pygame.time.Clock()
    title=pygame.font.SysFont("Microsoft YaHei",30,bold=True)
    body=pygame.font.SysFont("Microsoft YaHei",22)
    tile_font=pygame.font.SysFont("Microsoft YaHei",54,bold=True)
    agent.reset(seed)
    env=Game2048Env()
    grid,info=env.reset(seed=seed)
    # Compile before the visible event loop starts.
    agent.act(grid)
    agent.reset(seed)
    paused,finished,running=False,False,True
    colors={2:(238,228,218),4:(237,224,200),8:(242,177,121),16:(245,149,99),
            32:(246,124,95),64:(246,94,59),128:(237,207,114),256:(237,204,98),
            512:(237,200,80),1024:(237,197,63)}
    last=0
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    running=False
                elif event.key==pygame.K_SPACE:
                    paused=not paused
                elif event.key==pygame.K_r:
                    seed+=1
                    agent.reset(seed)
                    grid,info=env.reset(seed=seed)
                    finished=False
        now=pygame.time.get_ticks()
        if not paused and not finished and now-last>=1000/fps:
            action=agent.act(grid)
            if action<0:
                finished=True
            else:
                grid,_,finished,_,info=env.step(action)
            last=now
        screen.fill((250,248,239))
        def textline(text,y,font=body,color=(80,72,64)):
            screen.blit(font.render(text,True,color),(35,y))
        textline("3 x 3 2048 / "+policy_name,25,title)
        textline(f"Tile sum  {info['tile_sum']}       Maximum  {info['max_tile']}       Moves  {info['steps']}",78)
        status="TARGET 256 REACHED - continuing for more mass" if info["reached_target"] else "Goal: increase tile sum and reach 256"
        if finished and info['reached_target']:status="TARGET 256 REACHED / FINAL RESULT"
        textline(status,112,color=(42,120,88) if info["reached_target"] else (80,72,64))
        pygame.draw.rect(screen,(187,173,160),(35,160,650,650),border_radius=14)
        for r in range(3):
            for c in range(3):
                v=int(grid[r,c]); x=48+c*212; y=173+r*212
                pygame.draw.rect(screen,colors.get(v,(205,193,180) if not v else (70,65,60)),
                                 (x,y,200,200),border_radius=10)
                if v:
                    surf=tile_font.render(str(v),True,(119,110,101) if v<=4 else (255,255,255))
                    screen.blit(surf,surf.get_rect(center=(x+100,y+100)))
        message="GAME OVER" if finished else ("PAUSED" if paused else "RUNNING")
        textline(f"{message}   |   Space pause   R new game   Esc quit   |   Seed {seed}",819,font=pygame.font.SysFont("Microsoft YaHei",16))
        pygame.display.flip()
        if screenshot and finished:
            path=Path(screenshot)
            path.parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(screen,str(path))
            running=False
        clock.tick(60)
    env.close()
    pygame.quit()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent",choices=("mc","td","best"),default="best",help="mc=MCTS, td=TD, best=complete planning")
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--fps",type=float,default=8)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--games",type=int,default=1)
    parser.add_argument("--record",help="JSON replay path (first game)")
    parser.add_argument("--output",help="JSON per-game results path")
    parser.add_argument("--screenshot",help="Save the final game window as a PNG and exit")
    args=parser.parse_args()
    if args.games<1 or args.fps<=0:
        parser.error("games and fps must be positive")
    agent=Agent(args.agent,seed=args.seed)
    if not args.headless:
        interactive(agent,args.seed,args.fps,args.screenshot)
        return
    # Exclude initial JIT compilation from per-game times.
    agent.act(np.array([[2,0,0],[0,2,0],[0,0,0]]))
    results=[]
    for i in range(args.games):
        result,frames=episode(agent,args.seed+i,record=bool(args.record and i==0))
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if args.record and i==0:
            path=Path(args.record); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(model=agent.config,depth=agent.depth,result=result,frames=frames),ensure_ascii=False,indent=2),encoding="utf-8")
    if args.output:
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"{len(results)} games | mean sum {np.mean([r['tile_sum'] for r in results]):.2f} | >=256 {np.mean([r['reached_target'] for r in results]):.1%}")


if __name__=="__main__":
    main()
