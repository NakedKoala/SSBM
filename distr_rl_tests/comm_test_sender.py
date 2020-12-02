from ssbm_bot.distr_rl import communication

import time

import torch

pub_socket = communication.PairSocket(None, 50002, bind=True)

pub_socket.send(0, block=True)
print("done push")
print(pub_socket.recv())
print("done pull")
# time.sleep(10)

"""
for i in range(100):
    print(i)
    pub_socket.send(i)
    time.sleep(1)
"""

"""
time.sleep(3)
print("pub sending data")
pub_socket.send(torch.Tensor([1, 2, 3, 4, 5]), block=True)
pub_socket.send(torch.Tensor([6, 7, 8, 9, 10]), block=False)
pub_socket.send(torch.ones(256, 256))
print("pub done")
"""

"""
push_socket = communication.PushSocket(None, 50001)

print("push sending data")
push_socket.send(torch.Tensor([1, 2, 3, 4, 5]), block=False)
push_socket.send(torch.Tensor([6, 7, 8, 9, 10]), block=False)
push_socket.send(torch.ones(256, 256), block=False)
print("push done")
"""
