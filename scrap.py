import matplotlib.pyplot as plt
import time
import threading
import random

data = []

# This just simulates reading from a socket.
if __name__ == '__main__':
    # initialize figure
    fig = plt.figure()
    plt.ion()
    yy = []
    for i in range(5):
        data.append(random.random())
        yy.append(random.random())
        #plt.pause(1)
    #ln.set_xdata(range(len(data)))
    #ln.set_ydata(data)
    while True:
        fig.clear()
        plt.plot(data, yy, color='b')
        plt.draw()
        plt.show()
        plt.pause(1)
        data.append(random.random())
        yy.append(random.random())