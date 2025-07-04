#!/usr/bin/env python3
import os
import time
import numpy as np
# import cv2            # 미니맵 디코딩 시 필요
# import zmq            # ZMQ 미니맵 구독 시 필요
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy
)
from rclpy.executors import MultiThreadedExecutor
import threading
from gymnasium import Env, spaces

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleStatus,
    VehicleCommand,
    VehicleOdometry
)
# from nav_msgs.msg import Odometry  # 더 이상 사용하지 않음
# from nav_msgs.msg import OccupancyGrid  # /mini_map 토픽 수신 시 필요
from std_srvs.srv import Trigger

class PX4GymEnv(Node, Env):
    def __init__(self,
                 fixed_altitude: float = 5.0,
                 action_period: float = 0.1,
                 mode_rate_hz: float = 50.0,
                 minimap_size=(64,64),
                 zmq_address='tcp://localhost:5555'):
        super().__init__('px4_gym_env')

        #--- 파라미터 ---
        self.fixed_alt     = fixed_altitude
        self.action_period = action_period
        self.minimap_h, self.minimap_w = minimap_size

        # OffboardControlMode 퍼블리셔 QoS
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        # PX4 상태 토픽(Reliable/Volatile) 구독용 QoS
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        qos_sub_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        # 1) VehicleStatus 구독
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self._status_cb,
            qos_sub_be)

        # 2) VehicleOdometry 구독 (nav_msgs/Odometry 대신)
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._odom_cb,
            qos_sub_be)

        # 미니맵 토픽 구독 (추후 활성화)
        # self.map_sub = self.create_subscription(
        #     OccupancyGrid,
        #     '/mini_map',
        #     self._map_cb,
        #     qos_sub)

        # 제어용 퍼블리셔
        self.mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_pub)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_pub)
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_pub)

        # OffboardControlMode 지속 발행 타이머
        self.create_timer(1.0/mode_rate_hz, self._publish_offboard_mode)

        # 시뮬 재시작 서비스 클라이언트
        self.sim_restart = self.create_client(Trigger, '/sim/restart')
        if not self.sim_restart.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Simulation restart service unavailable')

        # ZMQ 미니맵 구독 초기화 (추후 활성화)
        # ctx = zmq.Context()
        # self.zmq_socket = ctx.socket(zmq.SUB)
        # self.zmq_socket.connect(zmq_address)
        # self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        # Gym 스페이스
        img_space   = spaces.Box(0,255,(self.minimap_h,self.minimap_w),np.uint8)
        state_space = spaces.Box(-np.inf,np.inf,(6,),np.float32)
        # state: [x, y, z, vx, vy, vz]
        self.observation_space = spaces.Dict({'image':img_space,'state':state_space})
        self.action_space      = spaces.Discrete(8)

        # 내부 상태
        self.current_status = None
        self.current_odom   = None
        self.map_data       = None

        # 백그라운드 spin
        #self.executor = MultiThreadedExecutor()
        #self.executor.add_node(self)
        #threading.Thread(target=self.executor.spin, daemon=True).start()

    # ─ 콜백 ─
    def _status_cb(self, msg: VehicleStatus):
        self.current_status = msg

    def _odom_cb(self, msg: VehicleOdometry):
        self.current_odom = msg

    # 미니맵 콜백 (추후 활성화)
    # def _map_cb(self, msg: OccupancyGrid):
    #     arr = np.array(msg.data, dtype=np.float32).reshape(
    #         (msg.info.height, msg.info.width))
    #     self.map_data = arr / 100.0  # 0~1 정규화

    def _publish_offboard_mode(self):
        m = OffboardControlMode()
        m.timestamp    = int(self.get_clock().now().nanoseconds/1000)
        m.position     = True
        m.velocity     = False
        m.acceleration = False
        self.mode_pub.publish(m)

    # ─ Gym API ─
    def reset(self, *, seed=None, options=None):
        # 1) 시뮬 재시작 요청
        req = Trigger.Request()
        fut = self.sim_restart.call_async(req)
        while not fut.done():
            print("waiting sim restart")
            time.sleep(0.1)
        if not fut.result().success:
            raise RuntimeError('Simulation restart failed: ' + fut.result().message)

        # 2) 상태·오도 대기
        while self.current_status is None or self.current_odom is None:
            print("waiting ros2 msg")
            time.sleep(1)

        # 3) 초기 Offboard 스트리밍 (10사이클)
        for _ in range(10):
            # TrajectorySetpoint: 고정 고도만
            sp = TrajectorySetpoint()
            sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            sp.position[2] = -self.fixed_alt
            self.traj_pub.publish(sp)
            time.sleep(0.1)
        print("sent init trajsetpoint")
        # 4) 모드 전환: OFFBOARD
        mode_cmd = VehicleCommand()
        mode_cmd.timestamp         = int(self.get_clock().now().nanoseconds / 1000)
        mode_cmd.command           = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        mode_cmd.param1            = 1.0   # main_mode flag
        mode_cmd.param2            = 6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        mode_cmd.target_system     = 1
        mode_cmd.target_component  = 1
        mode_cmd.source_system     = 1
        mode_cmd.source_component  = 1
        mode_cmd.from_external     = True
        self.cmd_pub.publish(mode_cmd)
        time.sleep(0.1)
        print("published offboard msg")
        # 5) 아밍(Arm)
        arm_cmd = VehicleCommand()
        arm_cmd.timestamp          = mode_cmd.timestamp
        arm_cmd.command            = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.param1             = 1.0   # arm
        arm_cmd.target_system      = 1
        arm_cmd.target_component   = 1
        arm_cmd.source_system      = 1
        arm_cmd.source_component   = 1
        arm_cmd.from_external      = True
        self.cmd_pub.publish(arm_cmd)
        print("sent arm command")
        # 6) OFFBOARD + ARMED 대기
        deadline = time.time() + 5.0
        while time.time() < deadline:
            s = self.current_status
            if (s.nav_state    == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                s.arming_state == VehicleStatus.ARMING_STATE_ARMED):
                break
            time.sleep(0.02)

        return self._build_obs(), {}

    def step(self, action):
        # OFFBOARD+ARMED 일 때만 명령 발행
        if (self.current_status.nav_state    == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
            self.current_status.arming_state == VehicleStatus.ARMING_STATE_ARMED):

            dx, dy = self._action_to_vector(action)
            t = TrajectorySetpoint()
            t.timestamp                   = int(self.get_clock().now().nanoseconds/1000)
            t.position[0], t.position[1], t.position[2] = dx, dy, -self.fixed_alt
            self.traj_pub.publish(t)
            print("published action:", dx, dy)
        else:
            print("skip publish: not OFFBOARD+ARMED")

        time.sleep(self.action_period)
        return self._build_obs(), 0.0, False, False, {}

    # 미니맵 수신 (추후 활성화)
    # def _recv_minimap(self, timeout=100):
    #     poll = zmq.Poller(); poll.register(self.zmq_socket, zmq.POLLIN)
    #     if dict(poll.poll(timeout)).get(self.zmq_socket):
    #         raw = self.zmq_socket.recv()
    #         img = cv2.imdecode(np.frombuffer(raw, np.uint8),
    #                            cv2.IMREAD_GRAYSCALE)
    #         return cv2.resize(img, (self.minimap_w, self.minimap_h))
    #     return np.zeros((self.minimap_h, self.minimap_w), np.uint8)

    def _build_obs(self):
        # 미니맵 (토픽 우선, 없으면 zeros)
        if self.map_data is not None:
            img = (self.map_data * 255).astype(np.uint8)
        else:
            img = np.zeros((self.minimap_h, self.minimap_w), np.uint8)

        # 상태 벡터: [x, y, z, vx, vy, vz]
        odom = self.current_odom
        pos = odom.position
        vel = odom.velocity
        st  = np.array([pos[0], pos[1], pos[2],
                        vel[0], vel[1], vel[2]],
                       dtype=np.float32)

        return {'image': img, 'state': st}

    def _action_to_vector(self, a):
        return {
            0: ( 1,  0), 1: (-1,  0),
            2: ( 0,  1), 3: ( 0, -1),
            4: ( 1,  1), 5: (-1,  1),
            6: ( 1, -1), 7: (-1, -1),
        }[a]

    def close(self):
        rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    env = PX4GymEnv()

    # 백그라운드 spin
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(env)
    threading.Thread(target=executor.spin, daemon=True).start()

    obs, _ = env.reset()
    for _ in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    env.close()
