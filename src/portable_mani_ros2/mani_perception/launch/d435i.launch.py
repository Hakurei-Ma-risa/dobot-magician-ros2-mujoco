from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Launch a D435i with the RGB-D contract consumed by Portable Mani."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_namespace", default_value="camera"),
            DeclareLaunchArgument("camera_name", default_value="camera"),
            DeclareLaunchArgument("color_profile", default_value="640x480x15"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            DeclareLaunchArgument("enable_sync", default_value="true"),
            DeclareLaunchArgument("align_depth", default_value="true"),
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace=LaunchConfiguration("camera_namespace"),
                name=LaunchConfiguration("camera_name"),
                output="screen",
                parameters=[
                    {
                        "enable_color": True,
                        "enable_depth": True,
                        "enable_infra": False,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "initial_reset": ParameterValue(
                            LaunchConfiguration("initial_reset"), value_type=bool
                        ),
                        "enable_sync": ParameterValue(
                            LaunchConfiguration("enable_sync"), value_type=bool
                        ),
                        "align_depth.enable": ParameterValue(
                            LaunchConfiguration("align_depth"), value_type=bool
                        ),
                        "enable_rgbd": False,
                        "pointcloud.enable": False,
                        "enable_accel": False,
                        "enable_gyro": False,
                        "color_qos": "SENSOR_DATA",
                        "depth_qos": "SENSOR_DATA",
                        "color_info_qos": "SENSOR_DATA",
                        "depth_info_qos": "SENSOR_DATA",
                        "rgb_camera.color_profile": LaunchConfiguration(
                            "color_profile"
                        ),
                        "depth_module.depth_profile": LaunchConfiguration(
                            "depth_profile"
                        ),
                    }
                ],
            ),
        ]
    )
