/* DY TEST */ // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
std::string server_ip = "169.254.84.7";  // 서버 IP
int server_port = 12345;

WSADATA wsaData;
SOCKET sock = INVALID_SOCKET;
struct sockaddr_in serv_addr;
char request[] = "GET_HAND";  // 바이너리로 요청할 데이터
char buffer[512] = { 0 };      // 수신 데이터를 저장할 버퍼

// Winsock 초기화
if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    std::cerr << "WSAStartup failed" << std::endl;
    break;
}

// 소켓 생성
if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
    std::cerr << "Socket creation error" << std::endl;
    WSACleanup();
    break;
}

serv_addr.sin_family = AF_INET;
serv_addr.sin_port = htons(server_port);

// 서버 IP 주소를 변환
if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
    std::cerr << "Invalid address or Address not supported" << std::endl;
    closesocket(sock);
    WSACleanup();
    break;
}

// 서버에 연결
if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cerr << "Connection failed" << std::endl;
    closesocket(sock);
    WSACleanup();
    break;
}

while (true) {
    // 서버에 'GET_HAND' 요청을 바이너리로 전송
    send(sock, request, strlen(request), 0);
    //std::cout << "GET_HAND request sent" << std::endl;

    // 서버로부터 데이터 수신
    int valread = recv(sock, buffer, 512, 0);
    double received_data[63];
    std::memcpy(received_data, buffer, 512);
    std::cout << "Recieved data:" << std::endl;

    for (int i=0; i<3; i++){
        std::cout << received_data[i] << " ";
    }
    std::cout << std::endl;

    if (valread <=0 ) {
        std::cerr << "Failed to read data from server" << std::endl;
        break;
    }
}

// 소켓 종료
closesocket(sock);
WSACleanup();  // Winsock 종료
/* DY TEST */ // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<