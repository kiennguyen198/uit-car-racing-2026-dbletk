"""TCP ACKs only on inherited client fd. No ucr_lib import/connect/send/recv."""
import os,select,socket,sys,time
fd=int(sys.argv[1]);port=int(sys.argv[2])
sock=socket.socket(fileno=fd)
assert sock.getpeername()==('127.0.0.1',port)
assert sock.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)==socket.SOCK_STREAM
control=sys.stdin.fileno();active=False;calls=0
print('ACK_WORKER_READY',flush=True)
try:
    while True:
        ready,_,_=select.select([control],[],[],.001 if active else .05)
        if ready:
            data=os.read(control,1024)
            if not data or b'X' in data:break
            active=data[-1:]==b'1'
        if active:
            sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_QUICKACK,1)
            calls+=1
finally:
    sock.close()  # Only this process's inherited descriptor.
    print('ACK_WORKER_CALLS='+str(calls),flush=True)
