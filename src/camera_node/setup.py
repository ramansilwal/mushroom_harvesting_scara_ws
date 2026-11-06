from setuptools import find_packages, setup
import os 

package_name = 'camera_node'

package_name = 'camera_node'
submodules1 = "camera_node/depth_anything_v2"
submodules2 = "camera_node/depth_anything_v2/util"
submodules3 = "camera_node/depth_anything_v2/dinov2_layers"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules1,submodules2,submodules3],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rsilwal',
    maintainer_email='rsilwal@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
   tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "camera_node=camera_node.two_camera:main",
            "depth_node=depth_anything.depth:main"
        ],
    },
)
