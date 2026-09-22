from glob import glob
from setuptools import find_packages, setup


package_name = "mani_mujoco"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "mujoco", "mani_core", "mani_dobot"],
    zip_safe=True,
    maintainer="hongjin",
    maintainer_email="hongjin@example.com",
    description="MuJoCo ROS 2 backend for Portable Mani",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_server = mani_mujoco.server:main",
            "closed_loop_evaluator = mani_mujoco.closed_loop_evaluator:main",
            "state_viewer = mani_mujoco.state_viewer:main",
        ],
    },
)
