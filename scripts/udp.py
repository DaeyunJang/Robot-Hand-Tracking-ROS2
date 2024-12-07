import queue
import socket
import multiprocessing as mp
import traceback
import json
import struct

import socket
import threading
import numpy as np

class Receiver(threading.Thread):
    def __init__(self, name, ip, port):
        threading.Thread.__init__(self)
        self.udp_name = name
        self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        self.receiver_socket.bind((ip, port))
        self.receiver_socket.settimeout(0.1)
        self.recv_msg = None
        self.recv_end = False

    def __str__(self):
        return self.udp_name

    def run(self):
        self._receive()

    def _receive(self):
        while True:
            if self.recv_end:
                self.receiver_socket.close()
                break

            try:
                data, addr = self.receiver_socket.recvfrom(8388608)

                if len(data) >= 1:
                    # print(data)
                    # print(addr)
                    self.recv_msg = data

            except socket.timeout:
                pass

            except Exception as e:
                print('receive error... please connect again')

    def get_recv_data(self):
        if self.recv_msg is not None:
            return_msg = self.recv_msg
            self.recv_msg = None
            return return_msg

        else:
            return None

    def close_receiver_socket(self):
        self.recv_end = True
        print('closing receiver socket...')


class Sender:
    def __init__(self, queue_handpose, queue_points):
        self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.queue_handpose = queue_handpose
        self.queue_points = queue_points
        self.handpose = None
        self.hand_points = np.zeros((21, 3))
        self.config = self.load_config('udp_config.json')
        # self.target_ip = self.config['host']
        self.target_ip = self.config['local_host']
        self.target_port = self.config['port']
        self.buffer_size = self.config['buffer_size']
        self.lock = threading.Lock()
        self.client = None

        print(f'{self.target_ip} / {self.target_port}')

    # def __str__(self):
    #     return self.udp_name

    # @staticmethod
    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def get_hand_landmarks(self):
        try:
            self.handpose = self.queue_handpose.get()
            self.hand_points = self.queue_points.get()
            # print(f'{self.hand_points}')
        except queue.Empty:
            # except mp.queues.Empty:
            #     print('Queue is empty')
            return

    def send_messages(self):
        serv_addr = (self.target_ip, self.target_port)
        while True:
            self.get_hand_landmarks()
            wl = self.handpose['world_landmarks']
            palm_pose = self.handpose['palm_pose_xyz_euler_ZYZ_robot']

            try:
                if self.handpose is None or self.handpose['index'] is None:
                    # Do not send
                    data = b'NONE' + b'\0' * (self.buffer_size - len('NONE'))
                    byte_data = data
                else:
                    # world_landmarks = self.handpose['world_landmarks']
                    # byte_data = world_landmarks.tobytes()
                    # print(f'xyzYZ: {palm_pose}')
                    if self.handpose['good_hand_error'] <= 0.0015:
                        send_data = struct.pack("ffffff", 1000*palm_pose[0], 1000*palm_pose[1], 1000*palm_pose[2], palm_pose[3], palm_pose[4], palm_pose[5])
                        self.sender_socket.sendto(send_data, serv_addr)
                        print(f'{send_data} / {type(send_data)}')
                    # print(f'{world_landmarks[0]} {world_landmarks[1]} {world_landmarks[2]}')
                # try:
                #     send_data = struct.pack("ffffff", wl[0][0], wl[0][1], wl[0][2], wl[1][0], wl[1][1], wl[1][2])
                #     self.sender_socket.sendto(send_data, serv_addr)
                # except Exception as e:
                #     print(f"Exception in broadcast: {e}")
                #     traceback.print_exc()
                    # raise
            except Exception as e:
                print(f"Exception in broadcast: {e}")
                traceback.print_exc()
                continue

    def close_sender_socket(self):
        self.sender_socket.close()

def start_udp(queue_handpose, queue_points):
    udp_sender = Sender(queue_handpose, queue_points)
    udp_sender.send_messages()

if __name__ == "__main__":
    queue_handpose = mp.Queue()
    queue_points = mp.Queue()
    udp_process = mp.Process(target=start_udp, args=(queue_handpose, queue_points))
    udp_process.start()
    udp_process.join()

