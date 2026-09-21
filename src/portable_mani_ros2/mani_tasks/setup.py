from setuptools import find_packages, setup


package_name = "mani_tasks"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="hongjin",
    maintainer_email="hongjin@example.com",
    description="Robot-independent pick and place task executors",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["pick_server = mani_tasks.ros_pick_server:main"],
    },
)
