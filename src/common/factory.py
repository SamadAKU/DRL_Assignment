import yaml

def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_env(app_cfg_path: str, reward_cfg_path: str, eval_mode: bool = False, render_human: bool = False):
    app = _load_yaml(app_cfg_path)
    reward_cfg = _load_yaml(reward_cfg_path)
    name = app.get("name")

    if name == "snake":
        from envs.snake_env import make_snake_env
        return make_snake_env(reward_cfg, eval_mode=eval_mode)

    if name == "pacman":
        from envs.pacman_env import make_pacman_env
        return make_pacman_env(reward_cfg, eval_mode=eval_mode, render_human=render_human)
