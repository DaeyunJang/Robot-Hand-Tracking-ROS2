import socket
import json
import threading
import struct
import numpy as np

class TCPClient:
    def __init__(self):
        self.config = self.load_config('tcp_config.json')
        self.host = self.config['host']
        self.port = self.config['port']
        self.buffer_size = self.config['buffer_size']
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hand_points = None

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def connect(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
            threading.Thread(target=self.thread_receive_data, daemon=True).start()
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def thread_receive_data(self):
        try:
            while True:
                self.client_socket.sendall(b'GET_HAND')

                data = self.client_socket.recv(self.buffer_size)
                print(f'data (byte={len(data)}: {data}')
                if not data:
                    continue

                if data.startswith(b'NONE'):
                    # print("No hands")
                    continue

                if len(data) != self.buffer_size:
                    print(f"Received data of unexpected size: {len(data)} (must be {self.buffer_size})")
                    continue
                # handpoint : (21,3) array -> 63 points -> (63 * 4 = 252) bytes
                points_num = 63
                points_byte_len = 252
                unpacked_data = struct.unpack(f'!{points_num}f', data[:points_byte_len])
                hand_points = np.array(unpacked_data).reshape((21, 3))
                print(f"Received data[hand points]: {hand_points}")

        except Exception as e:
            print(f"Exception in receive_data: {e}")
        finally:
            self.close()

    def send_data(self, data):
        try:
            self.client_socket.sendall(b'GET_HAND')
        except Exception as e:
            print(f"Exception in send_data: {e}")

    def close(self):
        self.client_socket.close()
        print(f'client[{self.client_socket}] closed')

if __name__ == "__main__":
    client = TCPClient()
    client.connect()
    print(f'TCP client on.')
    while True:
        try:
            pass
        except KeyboardInterrupt:
            client.close()

