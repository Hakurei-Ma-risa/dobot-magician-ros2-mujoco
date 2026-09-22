from setuptools import find_packages, setup


package_name = "mani_gemini"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="hongjin",
    maintainer_email="hongjin@example.com",
    description="Simulation-only Gemini Robotics ER proposal adapter",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gemini_er_probe = mani_gemini.probe:main",
            "gemini_er_sim_probe = mani_gemini.ros_sim_probe:main",
            "gemini_sim_chat = mani_gemini.sim_chat:main",
        ],
    },
)
