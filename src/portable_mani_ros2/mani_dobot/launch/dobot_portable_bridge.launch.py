from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="mani_dobot",
                executable="dobot_bridge",
                name="mani_dobot_bridge",
                output="screen",
                parameters=[
                    {
                        "group_name": "arm",
                        "end_effector_name": "gripper",
                        "base_frame": "magician_base_link",
                        "tip_frame": "TCP",
                    }
                ],
            )
        ]
    )

