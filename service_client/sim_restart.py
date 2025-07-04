#!/usr/bin/env python3
import os
import subprocess
import time
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class SimRestartService(Node):
    def __init__(self):
        super().__init__('sim_restart_service')
        self.create_service(Trigger, '/sim/restart', self.handle_restart)
        self.get_logger().info('Service /sim/restart ready')

        self.px4_dir = '/project/PX4-Autopilot'
        self.bin_dir = os.path.join(self.px4_dir, 'build/px4_sitl_default/bin')
        self.client  = os.path.join(self.bin_dir, 'px4-uxrce_dds_client')
        self.agent_ip   = os.getenv('PX4_AGENT_IP',   '172.18.0.3')
        self.agent_port = os.getenv('PX4_AGENT_PORT', '8888')

    def handle_restart(self, request, response):
        try:
            # 1) 기존 SITL/Gazebo 종료
            for proc in ('px4_sitl','gzserver','gzclient', 'gz sim'):
                subprocess.run(['pkill','-f',proc], check=False)

            # 2) make + SITL 실행 (로그가 터미널에 그대로 출력됩니다)
            popen = subprocess.Popen(
                ['bash','-lc','make px4_sitl gz_x500'],
                cwd=self.px4_dir,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            self.get_logger().info('make px4_sitl gz_x500 started...')
            # 잠깐 기다려서 SITL이 기동하도록
            time.sleep(15.0)

            # 3) uxrce_dds_client stop  (이 출력도 터미널에 보입니다)
            subprocess.run(
                [self.client, 'stop'],
                cwd=self.bin_dir,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True
            )
            time.sleep(1.0)

            # 4) uxrce_dds_client start
            subprocess.run(
                [
                    self.client, 'start',
                    '-t','udp',
                    '-p',self.agent_port,
                    '-h',self.agent_ip
                ],
                cwd=self.bin_dir,
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True
            )

            response.success = True
            response.message = 'SITL relaunched and XRCE client reconfigured'
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Command failed: {e}')
            response.success = False
            response.message = str(e)
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
            response.success = False
            response.message = str(e)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = SimRestartService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
