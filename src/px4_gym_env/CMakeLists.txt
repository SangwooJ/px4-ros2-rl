cmake_minimum_required(VERSION 3.5)
project(px4_gym_env)

# ament_cmake + rclpy
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Python 패키지 설치
ament_python_install_package(${PROJECT_NAME})

ament_package()
