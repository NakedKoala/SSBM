from ssbm_bot.distr_rl import communication

import time

sub_socket = communication.PairSocket(None, 50002)

# time.sleep(10)
print(sub_socket.recv(block=True))
sub_socket.send(1)
# time.sleep(5)
# print(sub_socket.recv(block=False))

"""
time.sleep(5)
print("sub receiving data")
print(sub_socket.recv(block=False))
print(sub_socket.recv(block=False))
print(sub_socket.recv(block=False).shape)
print(sub_socket.recv(block=False))
print("sub done")
"""

"""
pull_socket = communication.PullSocket(None, 50001, bind=True)

print("pull receiving data")
print(pull_socket.recv())
print(pull_socket.recv())
print(pull_socket.recv().shape)
print(pull_socket.recv(block=False))
print("pull done")
"""
