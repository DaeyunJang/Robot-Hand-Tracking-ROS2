import socket
import json
import threading

class TCPClient:
    def __init__(self):
        self.config = self.load_config('tcp_config.json')
        self.host = self.config['host']
        self.port = self.config['port']
        self.buffer_size = self.config['buffer_size']
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def connect(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
            threading.Thread(target=self.receive_data, daemon=True).start()
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def receive_data(self):
        try:
            while True:
                data = self.client_socket.recv(self.buffer_size)
                if not data:
                    break
                print(f"Received data: {data.decode('utf-8')}")
        except Exception as e:
            print(f"Exception in receive_data: {e}")
        finally:
            self.client_socket.close()

    def send_data(self, data):
        try:
            self.client_socket.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"Exception in send_data: {e}")

if __name__ == "__main__":
    client = TCPClient()
    client.connect()
    while True:
        message = input("Enter message to send: ")
        client.send_data(message)
