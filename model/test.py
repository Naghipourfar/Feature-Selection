import sys

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/23/18.
    Email : mn7697np@gmail.com
"""

# def f(name, lastname):
#     print(name, lastname)
#
#
# # if __name__ == '__main__':
# #     from multiprocessing import Pool
# #
# #     with Pool(2) as p:
# #         p.starmap(f, [(1, 1), (2, 2)])
# from tensorflow.python.client import device_lib
#
# # print(device_lib.list_local_devices())
#
# import pandas as pd
#
# a = pd.DataFrame([[1, 1], [2, 2], [3, 3]])
# print(a)
# b = [k for k in a.columns]
# print(b)
#
# print(type(a.values))

import matplotlib.pyplot as plt

# draw the figure so the animations will work
fig = plt.figure()
fig.show()
fig.canvas.draw()

while True:
    # compute something
    plt.plot([1], [2])  # plot something

    # update canvas immediately
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    # plt.pause(0.01)  # I ain't needed!!!
    fig.canvas.draw()
