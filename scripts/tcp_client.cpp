#include <iostream>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
    std::string server_ip = "127.0.0.1";  // 서버 IP (로컬 서버)
    int server_port = 12345;              // 서버 포트

    int sock = 0;
    struct sockaddr_in serv_addr;
    char request[] = "GET_HAND";  // 바이너리로 요청할 데이터
    char buffer[1024] = {0};      // 수신 데이터를 저장할 버퍼

    // 소켓 생성
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return 0;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(server_port);

    // 서버 IP 주소를 변환
    if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address or Address not supported" << std::endl;
        return 0;
    }

    // 서버에 연결
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        return 0;
    }

    while (true) {
    // 서버에 'GET_HAND' 요청을 바이너리로 전송
    send(sock, request, strlen(request), 0);

    // 서버로부터 데이터 수신
    int valread = read(sock, buffer, 1024);

    double received_data[63];
    std::memcpy(received_data, buffer, 512);
    std::cout << "Recieved data:" << std::endl;

    for (int i=0; i<3; i++){
        std::cout << received_data[i] << " ";
    }
    std::cout << std::endl;
    close(sock);
    return 0;
}
