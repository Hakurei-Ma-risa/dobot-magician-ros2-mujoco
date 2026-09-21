from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("viewer", default_value="false"),
            DeclareLaunchArgument("publish_camera", default_value="true"),
            DeclareLaunchArgument("camera_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("camera_width", default_value="320"),
            DeclareLaunchArgument("camera_height", default_value="240"),
            Node(
                package="mani_mujoco",
                executable="mujoco_server",
                name="mani_mujoco_server",
                output="screen",
                additional_env={"MUJOCO_GL": "egl"},
                parameters=[
                    {
                        "viewer": ParameterValue(
                            LaunchConfiguration("viewer"), value_type=bool
                        ),
                        "publish_camera": ParameterValue(
                            LaunchConfiguration("publish_camera"), value_type=bool
                        ),
                        "camera_rate_hz": ParameterValue(
                            LaunchConfiguration("camera_rate_hz"), value_type=float
                        ),
                        "camera_width": ParameterValue(
                            LaunchConfiguration("camera_width"), value_type=int
                        ),
                        "camera_height": ParameterValue(
                            LaunchConfiguration("camera_height"), value_type=int
                        ),
                    }
                ],
            ),
            Node(
                package="mani_perception",
                executable="rgbd_localizer",
                name="mani_rgbd_localizer",
                output="screen",
            ),
            Node(
                package="mani_tasks",
                executable="pick_server",
                name="mani_pick_server",
                output="screen",
            ),
        ]
    )
