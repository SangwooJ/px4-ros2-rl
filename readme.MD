docker run -it --privileged --name px4_simulatorn --network ros2-net --privileged -e ROS_DOMAIN_ID=0 -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix ununpentium/px4-gzbosim-humble:v1.0
  
  
  
docker run -it --privileged --name rl_agent --network ros2-net -p 8888:8888/udp -p 7400:7400/udp --gpus '"all"' -e ROS_DOMAIN_ID=0  ununpentium/ros2-px4agent-rlenv:v1.3




xhost +local: