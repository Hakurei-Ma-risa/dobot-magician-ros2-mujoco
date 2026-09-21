from gymnasium.utils.env_checker import check_env

from dobot_mujoco.env import DobotMagicianReachEnv


def test_gymnasium_contract() -> None:
    env = DobotMagicianReachEnv(max_episode_steps=5)
    check_env(env, skip_render_check=True)
    env.close()


def test_environment_steps() -> None:
    env = DobotMagicianReachEnv(max_episode_steps=5)
    observation, info = env.reset(seed=3)
    assert observation.shape == (14,)
    assert info["distance"] >= 0.0
    for _ in range(5):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert observation.shape == (14,)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    env.close()

