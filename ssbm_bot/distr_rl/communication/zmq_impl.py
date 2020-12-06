from .communication import *

import zmq

class ZMQSocketBase():
    def __init__(self, address, port, socket_type, bind=False, socket_opts=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(socket_type)
        if socket_opts:
            for opt in socket_opts:
                self.socket.setsockopt(*opt)
        if address is None:
            address = 'localhost'
        if bind:
            address = '*'
        full_addr = "tcp://%s:%s" % (str(address), str(port))
        if bind:
            self.socket.bind(full_addr)
        else:
            self.socket.connect(full_addr)

class ZMQSendSocket(ZMQSocketBase, SendSocketBase):
    def __init__(self, address, port, socket_type, bind=False, socket_opts=None):
        ZMQSocketBase.__init__(self, address, port, socket_type, bind=bind, socket_opts=socket_opts)

    def send_impl(self, data, block):
        if block:
            flags = 0
        else:
            flags = zmq.NOBLOCK
        try:
            self.socket.send(data, flags=flags)
        except zmq.ZMQError:
            return False
        return True

class ZMQPubSocket(ZMQSendSocket):
    def __init__(self, address, port, bind=False):
        super().__init__(address, port, zmq.PUB, bind=bind)

class ZMQPushSocket(ZMQSendSocket):
    def __init__(self, address, port, bind=False):
        super().__init__(address, port, zmq.PUSH, bind=bind)


class ZMQRecvSocket(ZMQSocketBase, RecvSocketBase):
    def __init__(self, address, port, socket_type, bind=False, socket_opts=None):
        ZMQSocketBase.__init__(self, address, port, socket_type, bind=bind, socket_opts=socket_opts)

    def recv_impl(self, block):
        if block:
            flags = 0
        else:
            flags = zmq.NOBLOCK
        try:
            return self.socket.recv(flags=flags)
        except zmq.ZMQError as e:
            return None

# NOTE: only the latest message is received.
class ZMQSubSocket(ZMQRecvSocket):
    def __init__(self, address, port, bind=False):
        super().__init__(address, port, zmq.SUB, bind=bind,
            socket_opts=(
                (zmq.SUBSCRIBE, b""),
                (zmq.CONFLATE, 1)
            )
        )

class ZMQPullSocket(ZMQRecvSocket):
    def __init__(self, address, port, bind=False):
        super().__init__(address, port, zmq.PULL, bind=bind)


class ZMQPairSocket(ZMQSendSocket, ZMQRecvSocket):
    def __init__(self, address, port, bind=False, socket_opts=None):
        ZMQSocketBase.__init__(self, address, port, zmq.PAIR, bind=bind, socket_opts=socket_opts)
