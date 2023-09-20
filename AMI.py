from optparse import AmbiguousOptionError
import numpy as np
import pandas as pd
import math
import statistics
from numpy import *
import csv
from scipy.stats import multivariate_normal
from scipy.stats import norm

class AMI_Thomas:
    def main():
        x = []
        L = 35

        file = 'fnnData.csv'

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(row.get('Row1'))

        def ami_body(x, L):
            #print(x)
            x = np.array(x, ndmin = 2)
            #print(x.shape)
            [m,n] = np.shape(x)
            if m ==1:
                x_row = pd.DataFrame(x)
                #print(x_row)
                x_row = x_row.T


            elif (m > 1 and n > 1):
                print('Input time series is not a one dimensional vector')

            AMI = np.zeros((L,2))
            for i in range(0, L):
                AMI[i][0] = i+1
                #print(AMI)
                x_list = x_row.values.tolist()
                #print(x_list[-1])

                newX = x_list[0:-1 - i]
                newX = pd.DataFrame(newX)

                newY = x_list[i+1:-1]
                newY.append(x_list[-1])


                newY = pd.DataFrame(newY)

                XYframe = pd.DataFrame(newX)
                XYframe = XYframe.assign(newColumn = newY)
                #print(AMI)
                AMI[i][1] = average_mutual_information(XYframe)
            #print(AMI)

            tau = []
            for i in range(1, len(AMI) - 1):
                if (AMI[i-1][1] >= AMI[i][1] and AMI[i][1] <= AMI[i+1][1]):
                    tau.append(AMI[i,:])
            #print(AMI[:,1])
            #print(0.2*AMI[0][1])

            ind = np.argwhere( (AMI[:,1] <= (0.2*AMI[0][1])) )
            #print(ind)
            if len(ind) == 0:
                tau.append(AMI[ind,:])
            
            if len(tau) == 0:
                if L*1.5 > len(x/3):
                    tau = 9999
                else:
                    print('Max lag needed to be increased for AMI_Thomas')
                    [tau, AMI] = ami_body(x, math.floor(L*1.5))
            print(tau)


        # Calculates average mutual information between two columns of data.
        # Uses kernel density estimation, with a globally adjusted Gaussian kernel.
        # Input: n-by-2 matrix, with data sets in adjecent column vectors
        # Output: scalar
        def average_mutual_information(data):
            n = len(data)
            #print(data)
            X = pd.DataFrame(data = data)
            Y = pd.DataFrame(data = data)
            #print(X)
            X = X.iloc[:,0]
            Y = Y.iloc[:,1]


            
            X = X.tolist()
            Y = Y.tolist()
            for i in range(0, len(X)):
                X[i] = int(X[i])

            for i in range(0, len(Y)):
                Y[i] = int(Y[i])

            #print(X)
            stdx = statistics.pstdev(X)
            stdy = statistics.pstdev(Y)
            #print(stdx)
            hx = ((stdx) / (n**(1/6)))
            hy = ((stdy) / (n**(1/6)))
            X = pd.DataFrame(X)
            Y = pd.DataFrame(Y)

            #hx = statistics.stdev(X) / (n**(1/6))
            #hy = statistics.stdev(Y) / (n**(1/6))

            P_x = univariate_kernel_density(X, X, hx)
            P_y = univariate_kernel_density(Y, Y, hy)

            JointP_xy = bivariate_kernel_density(data, data, hx, hy)

            multiplyFrames = P_x.mul(P_y)
            multiplyFrames = multiplyFrames.to_numpy()
            #print(multiplyFrames)
            JointP_xy = JointP_xy.to_numpy()
            #print(math.log2(JointP_xy / (multiplyFrames)))
            #print(sum(np.log2(JointP_xy / (multiplyFrames))))

            output = (sum(np.log2(JointP_xy / (multiplyFrames))) / n)
            return output
        
        def normpdf(x, mu, sigma):
            u = (x-mu)/abs(sigma)
            y = (1/(np.sqrt(2*math.pi)*abs(sigma)))*math.exp(-u*u/2)
            return y

        def univariate_kernel_density(value, Data, window):
            h = window
            valueShape = value.shape
            dataShape = Data.shape
            n = max(dataShape)
            m = max(valueShape)
            #n = len(Data)
            #m = len(value)

            prob = np.zeros((n,m))
            G = Extended(value, n, 0)
            

            newData = pd.DataFrame(data = Data)
            newData = newData.transpose()
            H = Extended(newData, m, 1)
            prob = norm.pdf((G-H) / h)
            #prob = normpdf( (G - H) / h, 0, 1 )
            #print(Prob)
            xstufft = [-2,-1, 0, 1, 2]
            b = [[-3,-2,-1], [-6,-5,-4], [-9,-8,-7]]
            #print(b)
            #print(norm.pdf(xstufft))
            fhat = np.sum(prob, axis = 0) / (n*h)
            y = pd.DataFrame(data = fhat)
            y = y.transpose()
            return y


        def bivariate_kernel_density(value, data, Hone, Htwo):
            s = np.shape(data)
            n = s[0]
            t = np.shape(value)
            number_pts = t[0]

            data[0] = data[0].astype('int')
            data["newColumn"] = data["newColumn"].astype('int')
            data = data.to_numpy()
            col1 = data[:,0]
            col2 = data[:,1]

            rho_matrix = np.corrcoef(col1, col2)
            rho = rho_matrix[0][1]

            W = [[Hone**2, rho*Hone*Htwo], [rho*Hone*Htwo, Htwo**2]]
            Differences = linear_depth(data, -data)

            # Does this work the same way as in matlab?
            mvn = multivariate_normal.pdf((Differences), [0,0], W)

            # Access probability as mvn.prob 
            Cumprob = np.cumsum(mvn)

            returnedProb = []
            returnedProb.append((1/n) * Cumprob[n])
            


            for i in range(1, number_pts):
                index = n*i
                returnedProb.append((1/(n)) * (Cumprob[index] - Cumprob[index - n]))
            
            outputMatrix = pd.DataFrame(data = returnedProb)
            outputMatrix = outputMatrix.transpose()
            
            return outputMatrix

        def linear_depth(feet, toes):
            if np.size(feet, 1) == np.size(toes, 1):
                a = np.size(feet, 0)
                b = np.size(toes, 0)

                Blocks = np.zeros((a*b, np.size(toes, 1)))
                Bricks = np.zeros((a*b, np.size(toes, 1)))

                for i in range(0, a):
                    Blocks[(i) * b: (i+1)*b,:] = feet[i]
                    Bricks[(i) * b: (i+1)*b,:] = toes
            return Blocks + Bricks


        def Extended(vector, n, trans):
            M = vector
            #print(vector)
            vectorList = vector
            M_list = []
            if trans == 0:
                vectorList = vector[0].tolist()
                for i in range(0, n):
                    M_list.append(vectorList)
                    i = i + 1
                M_list = np.array(M_list)
                M_list = M_list.T
            elif trans == 1:
                vectorList = vector.iloc[0].tolist()
                for i in range(0, n):
                    M_list.append(vectorList)
                    
                    i = i + 1
                M_list = np.array(M_list)
            return M_list
        
        ami_body(x, L)

    if __name__ == "__main__":
        main()