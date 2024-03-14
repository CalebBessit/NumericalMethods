import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    plt.show()


# Initial conditions
mu      = math.pow(10,-1)    #Initial shift from zero to prevent division by zero
otherBC = 0.0
y0      = [mu, otherBC]     # Initial values for y1 and y2
t_span  = [0+mu, 26]      # Time span for integration

def calculateMetric(y):
    sumTerms = 1
    start = 0
    metric = 0
    for k in range(sumTerms):
        start-=1
        metric += y[start]
    return metric


# #Bounds for one-node solution
lowerBound, upperBound = 2.16,2.185

# #Bounds for monotonically decaying solution
# lowerBound, upperBound = 1.5,1.52

epsilon                = math.pow(10,-6)    #Tolerance
N                      = 500                #Subintervals

#Initial conditions
y0      = [lowerBound, otherBC]
t_eval  = np.linspace(t_span[0], t_span[1], N+1)
sol     = solve_ivp(ode_system, t_span, y0, method='RK45',t_eval=t_eval)

# Extract the solution
y1, y2 = sol.y
metric = calculateMetric(y1)

#Testing whether lower value is good enough
if abs(metric)<epsilon:
    print("Error: {:.10f}".format(metric))
    print("Left BC: y({:.2f})={:.10f}".format(t_span[0], lowerBound))
    plotResults(sol)
    

#Same check for upper bound
y0 = [upperBound, otherBC]
t_eval = np.linspace(t_span[0], t_span[1], N+1)
sol = solve_ivp(ode_system, t_span, y0, method='RK45',t_eval=t_eval)
y1, y2 = sol.y
metric = calculateMetric(y1)

if abs(metric)<epsilon:
    print("Error: {:.10f}".format(metric))
    print("Left BC: y({:.2f})={:.10f}".format(t_span[0], upperBound))
    plotResults(sol)
    

#We need a higher accuracy than that provided by either of the values,
#So iterate using the bisection method to increase the accuracy
iterations  = 0
maxIter     = 100
while (abs(metric)>epsilon) and (iterations<maxIter):

    iterations +=1
    newVal      = (lowerBound+upperBound)/2
    print("Trying: {:.10f}".format(newVal))

    y0      = [newVal, otherBC]
    t_eval  = np.linspace(t_span[0], t_span[1], N+1)
    sol     = solve_ivp(ode_system, t_span, y0, method='RK45',t_eval=t_eval)
    y1, y2  = sol.y
    metric  = calculateMetric(y1)

    if abs(metric)<epsilon:
        print("Error: {:.10f}".format(metric))
        print("Left BC: y({:.2f})={:.10f}".format(t_span[0], upperBound))
        plotResults(sol)
 
    # # One node solution condition
    if metric>=0:
        upperBound = newVal
    elif metric<0:
        lowerBound = newVal

    #Monotonically decaying condition  
    # if metric>=0:
    #     lowerBound = newVal
    # elif metric<0:
    #     upperBound = newVal

if iterations>=maxIter:
    print("Maximum iterations exceeded. Stopping...")
    plotResults(sol)

