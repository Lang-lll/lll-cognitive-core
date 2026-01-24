import time
import threading
import requests


class SimpleHeartbeatClient:
    def __init__(self, server_url, register_msg, heartbeat_msg, check_interval=30):
        """
        server_url: 服务器URL
        check_interval: 检查间隔（秒）
        """
        self.server_url = server_url
        self.check_interval = check_interval
        self.is_registered = False
        self.last_heartbeat_time = 0
        self.is_running = False
        self.thread = None
        self.register_msg = register_msg
        self.heartbeat_msg = heartbeat_msg

    def start(self):
        """启动心跳线程"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        # FIXME: 服务还没启动
        self._run_fn()
        print(f"心跳检测启动，检查间隔：{self.check_interval}秒")

    def stop(self):
        """停止心跳线程"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("心跳检测停止")

    def _run(self):
        """主循环"""
        while self.is_running:
            try:
                self._run_fn()

                # 等待下次检查
                time.sleep(self.check_interval)

            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(5)  # 出错后等待5秒重试

    def _run_fn(self):
        # 太久没收到心跳转为未注册
        if time.time() - self.last_heartbeat_time > self.check_interval * 3:
            self.is_registered = False
            print(f"超时断开连接")

        # 判断是否已注册
        if not self.is_registered:
            # 发送注册消息
            self._send_register()
        else:
            # 发送心跳消息
            self._send_heartbeat()

    def _send_register(self):
        """发送注册消息"""
        try:
            print(f"发送注册请求: {self.register_msg}")

            requests.post(f"{self.server_url}", json=self.register_msg, timeout=5)

        except Exception as e:
            print(f"注册请求错误: {e}")

    def _send_heartbeat(self):
        """发送心跳消息"""
        try:
            print(f"发送心跳: {self.heartbeat_msg}")

            requests.post(f"{self.server_url}", json=self.heartbeat_msg, timeout=5)

        except Exception as e:
            print(f"心跳请求错误: {e}")

    def receive_registered(self):
        self.is_registered = True
        print(f"接收注册成功消息")

    def receive_heartbeat(self):
        self.last_heartbeat_time = time.time()
        print(f"接收心跳消息，时间: {self.last_heartbeat_time}")
