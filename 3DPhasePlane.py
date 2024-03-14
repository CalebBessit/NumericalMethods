#Plotting phase plane of system in three dimensions
#Caleb Bessit
#13 March 2024

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


# Define the system of ODEs
def ode_system(t, y):
    dydt = np.zeros_like(y)

    '''
        Note:
        y' = z = y[1]
        z' = f(t,y,z);
        y  = y[0]
    '''

    dydt[0] = y[1]
    dydt[1] = y[0]-2*y[0]**3-(1/t)*y[1]
    return dydt

def plotResults(sol):
    # Plot the results
    t = sol.t
    y1, y2 = sol.y
    plt.plot(t, y1, label='y(t)')
    # plt.plot(t, y2, label='y2(t)')
    plt.xlabel('r')
    plt.ylabel('Solution: v(r)')
    plt.title("Solution to v''+(1/r)v'-v+2v^3=0 ")
    plt.legend()
    plt.grid(True)
    print(y1[0])
    plt.show()


'''Initial plotting'''
# Initial conditions

mu      = math.pow(10,-1)    #Initial shift from zero to prevent division by zero
otherBC = 0.0
y0      = [mu, otherBC]     # Initial values for y1 and y2
t_span  = [0+mu, 25]      # Time span for integration

#Set of initial points to evaluate from
initConds = [0.5, 1.510310088167497,2.1802134570363885,3]
N         = 700

fig     = plt.figure()
ax      = plt.axes((0.05,0,0.9,0.9),projection='3d')
colors  = ["green","orange"]

for initCond in range(len(initConds)):
    y0 = [initConds[initCond], 0.0]
    t_eval = np.linspace(t_span[0], t_span[1], N+1)
    sol = solve_ivp(ode_system, t_span, y0, method='RK45',t_eval=t_eval)

    # Extract the solution
    y1, y2 = sol.y
    
    ax.plot(t_eval,y1,y2,color=colors[initCond])


ax.set_title("Phase plane", fontsize = 13)
ax.set_xlabel('r', fontsize = 11)
ax.set_ylabel('v(r)', fontsize = 11)
ax.set_zlabel("z(r)=v'(z)", fontsize = 10)    

plt.show()
    
