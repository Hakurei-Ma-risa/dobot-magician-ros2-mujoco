from glob import glob
from setuptools import find_packages, setup


package_name = "mani_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="hongjin",
    maintainer_email="hongjin@example.com",
    description="Course vision modules: find objects, choose target (rule or DeepSeek)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "find_objects = mani_vision.find_objects_node:main",
            "choose_target = mani_vision.choose_target_node:main",
            "pick = mani_vision.pick_cli:main",
        ],
    },
)
