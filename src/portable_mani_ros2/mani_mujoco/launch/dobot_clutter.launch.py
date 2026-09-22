"""Base-mounted RGB-D clutter scene with a separate MuJoCo viewer."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    source = PythonLaunchDescriptionSource(
        PathJoinSubstitution(
            [FindPackageShare("mani_mujoco"), "launch", "dobot_mujoco.launch.py"]
        )
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                source,
                launch_arguments={
                    "scene_mode": "clutter",
                    "viewer": "false",
                    "visualize": "true",
                    "publish_camera": "true",
                    "publish_perfect_detections": "false",
                    "publish_ground_truth": "false",
                    "publish_sim_state": "true",
                    "localizer_support_anchor": "bbox_bottom",
                    "localizer_mode": "roi_points",
                    "camera_width": "640",
                    "camera_height": "480",
                }.items(),
            )
        ]
    )
