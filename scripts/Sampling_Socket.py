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
    def __init__(self, name, ip, port):
        self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_name = name
        self.target_ip = ip
        self.target_port = port

    def __str__(self):
        return self.udp_name

    def send_messages(self, msg):
        serv_addr = (self.target_ip, self.target_port)
        try:
            self.sender_socket.sendto(msg, serv_addr)
            # serv_msg, addr = self.socket.recvfrom(M_SIZE)

            # self.recv_msg = serv_msg.decode(encoding="utf-8")

        except Exception as e:
            print(e)

    # def receive_messages(self, ip, port):

    def close_sender_socket(self):
        self.sender_socket.close()
