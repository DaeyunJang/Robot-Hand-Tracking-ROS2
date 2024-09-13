import queue
import socket
import multiprocessing as mp
import sys
import threading
import json
import time
import numpy as np
import struct
import traceback

from boost.mpi import world


class TCPServer:
    def __init__(self, queue_handpose, queue_points):
        self.queue_handpose = queue_handpose
        self.queue_points = queue_points
        self.handpose = None
        self.hand_points = np.zeros((21, 3))
        self.config = self.load_config('tcp_config.json')
        # self.host = self.config['host']
        self.host = self.config['local_host']
        self.port = self.config['port']
        self.buffer_size = self.config['buffer_size']
        self.lock = threading.Lock()
        self.client = None

        self.pps = 0
        self.count = 0
        self.s_t = time.time()

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def handle_client(self):
        print(f"[*] Handling client {self.client.getpeername()}")
        try:
            while True:
                # request_flag = False
                request_flag = self.client_callback()
                if request_flag:
                    self.get_hand_landmarks()
                    self.broadcast(self.client)
                else:
                    print(f'handle client fail')
                    break

                time.sleep(0.01)
        except Exception as e:
            print(f"Exception in handle_client: {e}")
            traceback.print_exc()
        finally:
            # print(f"[*] Client {self.client.getpeername()} disconnected.")
            self.client.close()

    def client_callback(self):
        request = self.client.recv(self.buffer_size)
        if not request:
            return False

        if request.decode('utf-8') == 'GET_HAND':
            return True


    def broadcast(self, client_socket):
        # t = self.handpose['world_landmarks']
        # print(f'handpose : {self.handpose}')
        if self.handpose is None or self.handpose['index'] is None:
            data = b'NONE' + b'\0' * (self.buffer_size - len('NONE'))
            byte_data = data
        else:
            world_landmarks = self.handpose['world_landmarks']
            byte_data = world_landmarks.tobytes()
            print(f'{world_landmarks[0]} {world_landmarks[1]} {world_landmarks[2]}')
        try:
            client_socket.sendall(byte_data)
        except Exception as e:
            print(f"Exception in broadcast: {e}")
            traceback.print_exc()
            raise

    def get_hand_landmarks(self):
        try:
            self.handpose = self.queue_handpose.get_nowait()
            self.hand_points = self.queue_points.get_nowait()
            # print(f'{self.hand_points}')
        except queue.Empty:
        # except mp.queues.Empty:
        #     print('Queue is empty')
            return

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)  # 하나의 연결만 처리

        try:
            while True:
                time.sleep(0.5)
                print(f"[*] Listening on {self.host}:{self.port}")
                client_socket, addr = server.accept()
                self.client = client_socket
                print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
                print(f"[*] client info : {self.client}")
                self.handle_client()
        except KeyboardInterrupt:
            print("Server shutting down.")
            traceback.print_exc()
        finally:
            server.close()

def start_tcp_server(queue_handpose, queue_points):
    tcp_server = TCPServer(queue_handpose, queue_points)
    tcp_server.start_server()

if __name__ == "__main__":
    queue_handpose = mp.Queue()
    queue_points = mp.Queue()
    tcp_process = mp.Process(target=start_tcp_server, args=(queue_handpose, queue_points))
    tcp_process.start()
    tcp_process.join()
