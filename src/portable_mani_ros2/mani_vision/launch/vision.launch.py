"""Modules 2 + 3: find_objects and choose_target (camera, localizer and robot run elsewhere)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("selector", default_value="rule", description="rule or deepseek"),
            DeclareLaunchArgument("target_frame", default_value="magician_base_link"),
            DeclareLaunchArgument("table_z_m", default_value="-1.0", description="-1 = estimate"),
            Node(
                package="mani_vision",
                executable="find_objects",
                output="screen",
                parameters=[
                    {
                        "target_frame": LaunchConfiguration("target_frame"),
                        "table_z_m": LaunchConfiguration("table_z_m"),
                    }
                ],
            ),
            Node(
                package="mani_vision",
                executable="choose_target",
                output="screen",
                parameters=[{"selector": LaunchConfiguration("selector")}],
            ),
        ]
    )
