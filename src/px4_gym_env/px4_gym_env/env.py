import time
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import Env, spaces
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleStatus
from nav_msgs.msg import Odometry
# from nav_msgs.msg import OccupancyGrid  # 미니맵 사용 시 필요 (추후 구현)
# import cv2  # 미니맵 디코딩 용 (추후 구현)
# import zmq  # ZMQ 구독용 (추후 구현)
from std_srvs.srv import Trigger

class PX4GymEnv(Node, Env):
    def __init__(self,
                 # zmq_address: str = 'tcp://localhost:5555',  # ZMQ 주소 (추후 구현)
                 # minimap_size: tuple = (64, 64),           # 미니맵 크기 (추후 구현)
                 fixed_altitude: float = 5.0,
                 action_period: float = 0.1,
                 mode_rate_hz: float = 50.0):
        super().__init__('px4_gym_env')
        self.fixed_alt = fixed_altitude
        self.action_period = action_period
        # self.minimap_h, self.minimap_w = minimap_size  # 추후 구현

        # ROS2 퍼블리셔/서브스크라이버 설정
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self._status_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/fmu/out/odometry', self._odom_cb, 10)
        # 미니맵 사용 시 활성화
        # self.map_sub = self.create_subscription(
        #     OccupancyGrid, '/mini_map', self._map_cb, 10)

        self.mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # 시뮬레이션 재시작 서비스 클라이언트
        self.sim_restart = self.create_client(Trigger, '/sim/restart')
        if not self.sim_restart.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Simulation restart service unavailable')

        # OffboardControlMode 지속 발행 타이머
        mode_period = 1.0 / mode_rate_hz
        self.create_timer(mode_period, self._publish_offboard_mode)

        # 미니맵 사용 시 활성화
        # ctx = zmq.Context()
        # self.zmq_socket = ctx.socket(zmq.SUB)
        # self.zmq_socket.connect(zmq_address)
        # self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        # 관측/액션 스페이스 정의
        # 이미지 스페이스에 빈 값 지정 (미니맵 미사용)
        img_space = spaces.Box(0, 0, (1,), np.uint8)  # 더미 스페이스
        state_space = spaces.Box(-np.inf, np.inf, (5,), np.float32)
        self.observation_space = spaces.Dict({'image': img_space, 'state': state_space})
        self.action_space = spaces.Discrete(8)

        self.current_status = None
        self.current_odom = None
        # self.map_data = None  # 미니맵 데이터 (추후 구현)

    # ROS2 콜백
    def _status_cb(self, msg):
        self.current_status = msg

    def _odom_cb(self, msg):
        self.current_odom = msg

    # 미니맵 사용 시 활성화
    # def _map_cb(self, msg):
    #     arr = np.array(msg.data, np.float32).reshape((msg.info.height, msg.info.width))
    #     self.map_data = arr / 100.0

    # OffboardControlMode 발행
    def _publish_offboard_mode(self):
        m = OffboardControlMode()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.position = True
        self.mode_pub.publish(m)

    # Gym 인터페이스
    def reset(self, *, seed=None, options=None):
        # 컨테이너A에 재시작 요청
        req = Trigger.Request()
        fut = self.sim_restart.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() and not fut.result().success:
            self.get_logger().error('Simulation restart failed: ' + fut.result().message)

        # 초기 오도메트리 대기
        rclpy.spin_once(self, timeout_sec=1.0)
        while self.current_odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self._build_obs(), {}

    def step(self, action):
        dx, dy = self._action_to_vector(action)
        t = TrajectorySetpoint()
        t.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        t.position[0], t.position[1], t.position[2] = dx, dy, -self.fixed_alt
        self.traj_pub.publish(t)

        time.sleep(self.action_period)
        rclpy.spin_once(self, timeout_sec=0.01)

        return self._build_obs(), self._compute_reward(), self._check_done(), False, {}

    def close(self):
        rclpy.shutdown()

    # 헬퍼
    # 미니맵 수신 (추후 구현)
    # def _recv_minimap(self, timeout=100):
    #     poll = zmq.Poller(); poll.register(self.zmq_socket, zmq.POLLIN)
    #     if dict(poll.poll(timeout)).get(self.zmq_socket):
    #         raw = self.zmq_socket.recv()
    #         img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    #         return cv2.resize(img, (self.minimap_w, self.minimap_h))
    #     return np.zeros((self.minimap_h, self.minimap_w), np.uint8)

    def _build_obs(self):
        # 미니맵 미사용: dummy image
        image = np.zeros((1,), np.uint8)
        st = np.zeros(5, np.float32)
        if self.current_odom:
            p = self.current_odom.pose.pose.position
            v = self.current_odom.twist.twist.linear
            q = self.current_odom.pose.pose.orientation
            yaw = 2 * np.arctan2(q.z, q.w)
            st = np.array([p.x, p.y, v.x, v.y, yaw], np.float32)
        return {'image': image, 'state': st}

    def _action_to_vector(self, a):
        m = {0:(1,0),1:(-1,0),2:(0,1),3:(0,-1),4:(1,1),5:(-1,1),6:(1,-1),7:(-1,-1)}
        return m[a]

    def _compute_reward(self):
        return 0.0

    def _check_done(self):
        return False

if __name__ == '__main__':
    rclpy.init()
    env = PX4GymEnv()
    obs, _ = env.reset()
    for _ in range(100):
        o, r, d, *_ = env.step(env.action_space.sample())
        if d:
            break
    env.close()