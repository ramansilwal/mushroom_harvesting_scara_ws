import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_file_name = '/home/cotrobot/mushroom_harvesting_scara_ws/resources/urdf/2160N0_urdf.urdf'
    urdf = os.path.join(
        get_package_share_directory('robot_control'),
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()


    lidar_package = 'sllidar_ros2'

    # 2. Hardcode the specific C1 launch file name
    lidar_launch_file = 'sllidar_c1_launch.py'

    # 3. Locate the installation directory dynamically
    package_dir = get_package_share_directory(lidar_package)

    # 4. Construct the path to the launch file
    launch_path = os.path.join(package_dir, 'launch', lidar_launch_file)

    # 5. Create the inclusion action
    included_lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path)
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf]),
        Node(
            package='robot_control',
            executable='run_robot',
            name='run_robot',
            output='screen',),

        Node(
            package='camera_node',
            executable='camera_node',
            name='camera_node',
            output='screen',),
           
        Node(
            package='gripper',
            executable='run_gripper',
            name='run_gripper',
            output='screen',),
        included_lidar_launch,
        Node(
            package='process_laser',
            executable='laser_process_node',
            name='laser_process_node',
            output='screen',),
        Node(
            package='RRT',
            executable='run_rrt',
            name='RRT',
            output='screen',),
        
            
        # Node(
        #     package='TOF',
        #     executable='run_tof',
        #     name='tof',
        #     output='screen',),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/cotrobot/.rviz2/default.rviz']),
        
            ]
            )