from glob import glob
from setuptools import find_packages, setup


package_name = "mani_perception"

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
    description="Robot-independent RGB-D camera launch and diagnostics for Portable Mani",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_healthcheck = mani_perception.camera_healthcheck:main",
            "rgbd_localizer = mani_perception.rgbd_localizer:main",
        ],
    },
)
