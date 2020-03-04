import socket
import time

clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSock.connect(('localhost', 7788))

s = time.time()
sendDataLen = clientSock.send("recognize /home/robot/historyData/yyyyyyyy/colorful/xxxxxxxx/xxxxxxxx_colorful SecurityControl".encode())
recvData = clientSock.recv(1024)
print("sendDataLen: ", sendDataLen)
print("recvData: ", recvData)
print("Success! Time using {:.2f}".format(time.time() - s))
