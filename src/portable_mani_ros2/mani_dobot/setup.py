from glob import glob
from setuptools import find_packages, setup


package_name = "mani_dobot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "mani_core", "mani_tasks"],
    zip_safe=True,
    maintainer="hongjin",
    maintainer_email="hongjin@example.com",
    description="Dobot Magician adapter for Portable Mani",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dobot_bridge = mani_dobot.ros_bridge:main",
            "mujoco_reach_demo = mani_dobot.mujoco_demo:main",
            "mujoco_pick_eval = mani_dobot.pick_evaluator:main",
        ],
    },
)
