import argparse
from stable_baselines3 import PPO, A2C
from src.common.factory import make_env

ALGOS = {"ppo": PPO, "a2c": A2C}


def _load(algo: str, path: str):
    a = algo.lower()
    if a == "ppo": return PPO.load(path)
    if a == "a2c": return A2C.load(path)
    raise ValueError("algo must be ppo or a2c")


def watch(app: str, algo: str, model_path: str, episodes: int):
    env = make_env(app, for_watch=True, reward_cfg_path=None)
    model = _load(algo, model_path)
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += float(reward)
            env.render()
        print(f"[WATCH] {app} ep_return={ep_rew:.2f}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["pacman", "snake"])
    ap.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    watch(args.app, args.algo, args.model_path, args.episodes)


if __name__ == "__main__":
    main()
