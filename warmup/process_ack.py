"""Independent ACK option process, never another organizer API connection."""
import os,select,socket,subprocess,sys
from pathlib import Path

class ProcessAck:
    def __init__(self,port=11000):
        self.port=port;self.process=None;self.failed=False
        self.info=dict(enabled=False,ready=False,peer=None,original_fd=None,error=None)

    def begin(self):
        if self.failed:return
        if self.process is None:
            for name in os.listdir('/proc/self/fd'):
                try:
                    if not os.readlink('/proc/self/fd/'+name).startswith('socket:'):continue
                    sock=socket.fromfd(int(name),socket.AF_INET,socket.SOCK_STREAM)
                except (OSError,ValueError):continue
                try:
                    peer=sock.getpeername()
                    if peer[0]=='127.0.0.1' and peer[1]==self.port:
                        self.process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('ack_worker_process.py')),name,str(self.port)],pass_fds=(int(name),),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=False)
                        self.info.update(enabled=True,peer=list(peer),original_fd=int(name))
                        break
                except OSError:pass
                finally:sock.close()
        if self.process is not None:
            try:
                if self.process.poll() is not None:raise RuntimeError('ACK worker stopped')
                self.process.stdin.write(b'1');self.process.stdin.flush()
                if not self.info['ready'] and select.select([self.process.stdout],[],[],0)[0]:
                    self.info['ready']=self.process.stdout.readline().strip()==b'ACK_WORKER_READY'
            except (OSError,RuntimeError) as error:
                self.info.update(enabled=False,error=str(error));self.failed=True

    def pause(self):
        if self.process is not None and self.process.poll() is None:
            try:self.process.stdin.write(b'0');self.process.stdin.flush()
            except OSError:self.failed=True;self.info['enabled']=False

    def close(self):
        process=self.process
        if process is None:return
        try:
            if process.poll() is None:
                process.stdin.write(b'X');process.stdin.flush();process.stdin.close()
                process.wait(timeout=1)
        except (OSError,subprocess.TimeoutExpired):
            # Only our worker, never the car bot or map.
            process.terminate();process.wait(timeout=1)
        output=process.stdout.read().decode();errors=process.stderr.read().decode()
        self.info.update(worker_output=output,worker_errors=errors,worker_exit=process.returncode)
        self.process=None
