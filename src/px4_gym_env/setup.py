from setuptools import setup

package_name = 'px4_gym_env'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='PX4+Gazebo 기반 강화학습 환경',
    license='Apache-2.0',
    entry_points={
        # 필요하다면 console_scripts 여기에 등록
        # 'console_scripts': [
        #     'run_env = px4_gym_env.env:main'
        # ],
    },
)
