"""

Path tracking simulation with pure pursuit steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy import io
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import os

k = 0.1  # look forward gain
Lfc = 1.0  # look-ahead distance
Kp = 1.0  # speed propotional gain
dt = 0.1  # [s]
L = 1.04  # [m] wheel base of vehicle


show_animation = True
save_gif = True
class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


def pure_pursuit_control(state, cx, cy, pind):

    nearest_ind, ind = calc_target_index(state, cx, cy)
    
    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    if state.v < 0:  # back
        alpha = math.pi - alpha

    Lf = k * state.v + Lfc

    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind, nearest_ind


def calc_target_index(state, cx, cy):

    # search nearest point index
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    nearest_ind = d.index(min(d))
    L = 0.0
    ind = nearest_ind
    Lf = k * state.v + Lfc

    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cx[ind + 1] - cx[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1
    way_point = ind
    return nearest_ind, way_point




def main(cx, cy, target_speed, T, state):

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    
    
    nearest_ind, target_ind = calc_target_index(state, cx, cy)
    near = [nearest_ind] 
    target = [target_ind]
    
    
    while T >= time and lastIndex > target_ind:
        ai = PIDControl(target_speed, state.v)
        di, target_ind, nearest_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, ai, di)
        
        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        
        near.append(nearest_ind)
        target.append(target_ind)
        
        if show_animation:
            plt.cla()
            path_plot = plt.plot(cx, cy, "-r", linewidth=0.5, alpha=0.7, label="Path")
            erp_plot = plt.scatter(x=x[-1], y=y[-1], 
                                marker=MarkerStyle(">", transform=Affine2D().rotate_deg(yaw[-1]*45), fillstyle='none'),
                                # marker=(1, 0, yaw[-1]), #
                                s=50, 
                                linewidths=0.5,
                                color='blue')
            plt.text(x=x[-1], y=y[-1], s=f'{yaw[-1]*45:.2f}', fontsize=20)
            nearest_plot = plt.scatter(x=cx[nearest_ind], 
                                        y=cy[nearest_ind], 
                                        s=100,
                                        linewidths=0.5,
                                        marker=MarkerStyle(".", fillstyle='none'), 
                                        color='blue')
            target_plot = plt.scatter(x=cx[target_ind], 
                                        y=cy[target_ind], 
                                        marker=MarkerStyle("*", fillstyle='none'),
                                        linewidths=0.5,
                                        s=50,
                                        color='pink')

            #plt.grid(True)
            plt.xlim([-5, 5])
            
            plt.xticks(np.arange(-5, 6))
            plt.yticks(np.arange(-10, 80, step=10))
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.title('Path & ERP 42& Nearest Point & Waypoint Plot')
            
            plt.legend(handles=[path_plot, erp_plot, nearest_plot, target_plot], 
                        labels=["Path","ERP42", "Nearest Point", "Waypoint"], 
                        loc='upper right',
                        scatterpoints=1)
            plt.pause(0.001)
            
            # plt.show()

    if save_gif:
        fig = plt.figure()
        ani = matplotlib.animation.FuncAnimation(fig, save_animation, frames=50, interval=100)
        ani.save('animation.gif', writer='imagemagick', fps=30)

def save_animation(x, y, cx, cy, near, target, yaw):
    plt.cla()
    path_plot = plt.plot(cx, cy, "-r", linewidth=0.5, alpha=0.7, label="Path")
    erp_plot = plt.scatter(x=x, y=y, 
                        marker=MarkerStyle(">", transform=Affine2D().rotate_deg(yaw[-1]*45), fillstyle='none'),
                        # marker=(1, 0, yaw[-1]), #
                        s=50, 
                        linewidths=0.5,
                        color='blue')
    plt.text(x=x[-1], y=y[-1], s=f'{yaw[-1]*45:.2f}', fontsize=20)
    nearest_plot = plt.scatter(x=near, 
                                y=cy[nearest_ind], 
                                s=100,
                                linewidths=0.5,
                                marker=MarkerStyle(".", fillstyle='none'), 
                                color='blue')
    target_plot = plt.scatter(x=cx[target_ind], 
                                y=cy[target_ind], 
                                marker=MarkerStyle("*", fillstyle='none'),
                                linewidths=0.5,
                                s=50,
                                color='pink')

    #plt.grid(True)
    plt.xlim([-5, 5])
    
    plt.xticks(np.arange(-5, 6))
    plt.yticks(np.arange(-10, 80, step=10))
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title('Path & ERP 42& Nearest Point & Waypoint Plot')
    
    plt.legend(handles=[path_plot, erp_plot, nearest_plot, target_plot], 
                labels=["Path","ERP42", "Nearest Point", "Waypoint"], 
                loc='upper right',
                scatterpoints=1)
    


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    mat = io.loadmat('ERP42&Path.mat')
    #  target course
    cy = np.array(mat['Y'][:, 0])
    print(cy)
    cx = np.array(mat['x'][0])
    
    # target_speed = 10.0 / 3.6  # [m/s]
    
    # max simulation time
    T = cx.shape[0] 
    # initial state
    state = State(x=-5, y=36, yaw=0.0, v=0.0)
    main(cx=cx, cy=cy, target_speed=15, T=T, state=state)
