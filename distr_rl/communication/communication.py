import io
import torch

class TorchSocket(object):
    @staticmethod
    def to_bytes(data):
        buff = io.BytesIO()
        torch.save(data, buff)
        return buff.getvalue()

    @staticmethod
    def from_bytes(byte_data):
        buff = io.BytesIO(byte_data)
        return torch.load(buff)

class RecvSocketBase(TorchSocket):
    def __init__(self, address, port, bind=False):
        raise NotImplementedError()

    # return None if nothing was received
    def recv(self, block=True):
        byte_data = self.recv_impl(block)
        if byte_data:
            return self.from_bytes(byte_data)
        return None

    # return None if nothing was received
    def recv_impl(self, block):
        raise NotImplementedError()

class SendSocketBase(TorchSocket):
    def __init__(self, address, port, bind=False):
        raise NotImplementedError()

    # return True if send successful, False if not
    def send(self, data, block=True):
        return self.send_impl(self.to_bytes(data), block)

    def send_impl(self, data, block):
        raise NotImplementedError()
