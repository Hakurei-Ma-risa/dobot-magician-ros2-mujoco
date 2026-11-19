"""Whole course chain in simulation:

camera (MuJoCo D435i) -> find_objects -> choose_target -> rgbd_localizer
-> pick task -> MuJoCo robot.   Then:  ros2 run mani_vision pick "the red can"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("mani_mujoco"), "launch", "dobot_mujoco.launch.py"])
        ),
        launch_arguments={
            "scene_mode": "clutter",
            "viewer": LaunchConfiguration("viewer"),
            "visualize": "false",
            "publish_camera": "true",
            "publish_perfect_detections": "false",
            "publish_ground_truth": LaunchConfiguration("ground_truth"),
            "publish_sim_state": "true",
            "localizer_support_anchor": "bbox_bottom",
            "localizer_mode": "roi_points",
            "camera_width": "640",
            "camera_height": "480",
        }.items(),
    )
    vision = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("mani_vision"), "launch", "vision.launch.py"])
        ),
        launch_arguments={"selector": LaunchConfiguration("selector")}.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("selector", default_value="rule", description="rule or deepseek"),
            DeclareLaunchArgument("viewer", default_value="true"),
            DeclareLaunchArgument("ground_truth", default_value="false", description="for grading only"),
            sim,
            vision,
        ]
    )
