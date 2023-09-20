'''
xSE = Ent_xSamp20180320(x,y,m,R,norm)
% Inputs - x, first data series
%        - y, second data series
%        - m, vector length for matching (usually 2 or 3)
%        - R, R tolerance to find matches (as a proportion of the average 
%             of the SDs of the data sets, usually between 0.15 and 0.25)
%        - norm, normalization to perform
%          - 1 = max rescale/unit interval (data ranges in value from 0 - 1
%            ) Most commonly used for RQA.
%          - 2 = mean/Zscore (used when data is more variable or has 
%            outliers) normalized data has SD = 1. This is best for cross 
%            sample entropy.
%          - Set to any value other than 1 or 2 to not normalize/rescale 
%            the data
% Remarks
% - Function to calculate cross sample entropy for 2 data series using the
%   method described by Richman and Moorman (2000).
% Sep 2015 - Created by John McCamley, unonbcf@unomaha.edu
% Copyright 2020 Nonlinear Analysis Core, Center for Human Movement
% Variability, University of Nebraska at Omaha
%
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright 
%    notice, this list of conditions and the following disclaimer in the 
%    documentation and/or other materials provided with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its 
%    contributors may be used to endorse or promote products derived from 
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%
% Check both sets of data are the same length
'''
from telnetlib import BM
import numpy as np
import pandas as pd
import csv

class normalization:

    def main():

        firstData = []
        secondData = []
        file = 'testdata.csv'

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                firstData.append(row.get('firstdata'))
                secondData.append(row.get('seconddata'))

        vectorLength = 2
        rTolerance = 0.2
        norm = 1

        def calculate(firstData, secondData, vectorLength, rTolerance, norm):
            x1 = len(firstData)
            y1 = len(secondData)
            xn = []
            yn = []

            for i in range(0, x1):
                firstData[i] = float(firstData[i])
            for i in range(0, y1):
                secondData[i] = float(secondData[i])

            if x1 != y1:
                print('The data series need to be the same length!')

            N = len(firstData)

            if norm == 1:
                for i in range(0, x1):
                    xn.append((firstData[i] - min(firstData)) / (max(firstData) - min(firstData)))
                for j in range(0, y1):
                    yn.append((secondData[j] - min(secondData)) / (max(secondData) - min(secondData)))

                r = rTolerance * ((np.std(xn) + np.std(yn))/2)
                print(r)
                
                

            #  normalize data to have a SD = 1, and mean = 0
            elif norm == 2:
                for i in range(0, x1):
                    xn.append((firstData[i] - np.mean(firstData))/np.std(firstData))
                    yn.append((secondData[i] - np.mean(secondData))/np.std(secondData))
                
                r = rTolerance

            else:
                print('These data will not be normalized')
            
            m = 0

            for i in range(0, len(xn)):
                xn[i] = float(xn[i])
                
            for i in range(0, len(yn)):
                yn[i] = float(yn[i])

            dij = pd.DataFrame(0.0, index = range(N-vectorLength), columns = range(vectorLength+1))

            Bm = []
            Am = []
            d = []
            for i in range(0, (N - vectorLength)):
                for k in range(0, (vectorLength+1)):
                    temp_list = xn[k:N-vectorLength+k]
                    temp_list = list(map(lambda x: abs(x - yn[i+k]), temp_list))
                    dij[k] = temp_list

                sub_dij = dij.iloc[:,0:2]
                dj = np.amax(sub_dij, axis=1)
                dj1 = np.amax(dij, axis = 1)

                dj = dj.to_numpy()
                dj1 = dj1.to_numpy()


                d = np.argwhere(dj <= r)
                d1 = np.argwhere(dj1 <= r)

                nm = len(d)
                Bm.append(nm/(N-vectorLength))
                nm1 = len(d1)
                Am.append(nm1/(N-vectorLength))

            Bmr = sum(Bm)/(N-vectorLength)
            Amr = sum(Am)/(N-vectorLength)
            xSE = -np.log(Amr/Bmr) 
            print(xSE)
        
        calculate(firstData, secondData, vectorLength, rTolerance, norm)

    if __name__ == "__main__":
        main()
