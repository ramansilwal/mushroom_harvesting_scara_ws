from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'robot_control'
submodules = "robot_control/Hitbot"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.STL')),
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
            "run_robot=robot_control.robot_interface:main",  

        ],
    },
)
