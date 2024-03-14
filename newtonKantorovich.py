import numpy as np
import math
import matplotlib.pyplot as plt

'''Setup - Initial Parameters'''

a, b            = 0.01, 30                    #Left and right endpoints
alpha, beta     = 0, 0                        #Values at left and right endpoints

N               = 100                         #Number of subintervals
mu              = (b-a)/N                     #Width of subinterval
tolerance       = math.pow(10,-6)
maxIterations   = 100

#Domain
x   = np.linspace(a,b,N+1)
print(x)

#Intial guess
def generateInitialExp():
    global x
    d1, c1, f1, g1 = 3.4, 0.605, -2, 0.3
    y1 = d1*np.exp(-((c1*x)**2)) + f1*np.exp(-((g1*x)**3))
    #The derivative, in case it needs to be plotted
    # y2 = -2*(c1**2)*d1*x*np.exp(-((c1*x)**2)) + -3*(g1**3)*(x**2)*f1*np.exp(-((g1*x)**3))
    return y1.copy()


def generateInitialMono():
    global x
    a1, b1 = 1.51, 1
    y1 = a1*np.exp(-((b1*x)**2))
    return y1.copy()

#Generate initial condition: comment out line 36 for the monotonically decaying solution, comment out line 37 for the one-node solution
# y = generateInitialExp()
y = generateInitialMono()
y_0 = y.copy()

def constructJacobian(y):
    global mu,x,a
    #Construct the Jacobian. 
    #The middle (N-1)x(N-1) matrix
    lowerDiag, midDiag, upperDiag = [],[],[]
    #Despite the indices, this is for equation with subscript 2 -> N
    #0->2, and N
    for i in range(1,N):
        lowerDiag.append(1/(mu**2) - ( 1/(2*mu*(a+(i*mu))) ))
        upperDiag.append(1/(mu**2) + ( 1/(2*mu*(a+(i*mu))) ))
        value =  (-(2/(mu**2)) - 1 + 6*((y[i])**2))
        midDiag.append(value)


    lowerDiag, midDiag, upperDiag = np.diag(lowerDiag), np.diag(midDiag), np.diag(upperDiag)
    rows, columns   = len(lowerDiag), len(lowerDiag)+2
    zeroColumn      = np.zeros((rows,1))

    #Create Jacobian
    lowerDiag   = np.hstack((lowerDiag,zeroColumn,zeroColumn))
    midDiag     = np.hstack((zeroColumn,midDiag,zeroColumn))
    upperDiag   = np.hstack((zeroColumn,zeroColumn,upperDiag))
    jacobian    = lowerDiag + midDiag + upperDiag

    zeroElements    = columns-3
    topRow          = np.append(np.array([(-3/(2*mu)), (2/mu), (-1/(2*mu))]),np.zeros(zeroElements))
    bottomRow       = np.append(np.zeros(zeroElements), np.array([0,0,1]))
    jacobian        = np.vstack((topRow,jacobian, bottomRow))

    return jacobian

def constructRHS(y):
    global mu, alpha, beta,a
    #Calculate rhs

    firstElement = -1*(  -3*y[0] +4*y[1]-y[2]   )/(2*mu )
    lastElement  = -1*y[-1]

    rhs = []
    for i in range(1,N):
        value = -1* (  ( (y[i-1]-2*y[i]+y[i+1])  / (mu**2)  ) + ( (y[i+1]-y[i-1]) / (2*(a+(i*mu))*mu)  ) - y[i] +2*(y[i])**3  )
        rhs.append(value)

    rhs = [firstElement] + rhs + [lastElement]
    rhs = np.array(rhs)
    return rhs


jacobian = constructJacobian(y)
rhs      = constructRHS(y)

#Begin iterations
iterations = 0
z = np.linalg.solve(jacobian,rhs)

while (np.linalg.norm(rhs)>tolerance) :
    #Update guess
    iterations+=1
    y += z
    
    jacobian = constructJacobian(y)
    rhs      = constructRHS(y)
    z = np.linalg.solve(jacobian,rhs)

#Plotting initial guess
plt.figure("solutions")
plt.xlabel("r")
plt.ylabel("Solution: v(r)")
plt.title("Solution to v''+(1/r)v'-v+2v^3=0 using the Newton-Kantorovich method")

plt.plot(x,y_0, label="Initial guess")
plt.plot(x,y,   label="Solution after max iterations or within tolerance")
plt.grid(True)
plt.legend()

# #Plotting the derivative of the solution as well
# plt.figure("phase plane")
# plt.plot(y,yDiff)
# plt.grid(True)
plt.show()