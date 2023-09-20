# Coversion for FNN2020
from re import sub
import numpy as np
import pandas as pd
import math
import csv


class F_Nearest_Neighbors:


    def main():

        

        Rtol = 15  #threshold for first criterion
        Atol = 2  #threshold for second criterion
        speed = 0 #a 0 for the code to calculate to the MaxDim or a 1 for the code to finish once a minimum is found
        MaxDim = 4  #maximum embedded dimension
        data = []  #column oriented time series
        tau = 2  #time delay

        file = 'fnnData.csv'

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row.get('Row1'))



        def FNN2020(data, tau, MaxDim, Rtol, Atol, speed):

            for i in range(0, len(data)):
                data[i] = float(data[i])

            n = len(data)-tau*MaxDim  # of data points to be used
            RA = np.std(data)

            z = data[0:n]
            y_frame = pd.DataFrame(z)
            y = []
            #print(y)

            m_search = 2
            indx = list(range(0, n+1))
            indx = pd.DataFrame(indx)
            dim = []

            dE = np.zeros((MaxDim, 1))
            global pqd
            global pqr
            global pqz
            pqd = []
            pqr = []
            pqz = []

            for j in range(1,MaxDim+1):
                #print(j)
                y_frame = pd.DataFrame(z)
                if len(y) == 0:
                    y = y_frame
                else:
                    name = str((j-1))
                    y[name] = y_frame

                #print(y)

                #z = np.arange(data[(1+tau*j)-1], data[(n+tau*j)-1])
                z = data[(1+tau*j)-1:(n+tau*j)]
                L = np.zeros((n, 1))

                kd_part(y,z,512)
                #print(indx.iloc[0])
                for i in range(0, len(indx) -1):
                    #print(indx[i])
                    #print(y.iloc[i])
                    #print(i)
                    yq = y.iloc[indx.iloc[i]]
                    #print(yq)
                    #print("break")

                    b_upper = float('inf')*np.ones(np.shape(yq))
                    b_lower = -b_upper

                    #pqd = float('inf')*np.ones((m_search, 1))
                    
                    pqd = []
                    pqr = []
                    pqz = []
                    L_done = 0

                    #[pqd, y_model, z_model, temp1, temp2, pqz, temp3, temp4, sort_list, node_list] = kdSearch(
                        #1, m_search, yq, pqd, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)
                    
                    kdSearch(1, m_search, yq, y_model, z_model, L_done, b_upper, b_lower, sort_list, node_list)
                    #print(pqz)
                    
                    #print(pqz[0])
                    #new = []
                    
                    #for i in range(0, m_search):
                        #new.append(pqz[0][i])
                    #print(pqz)
                    #print(pqz[0])

                    #pqz[0] = pqz[0].astype('int')
                    #print(pqz)

                    if isinstance(pqz, pd.DataFrame):
                        print("hello")
                        distance = pqz.iloc[0] - pqz.iloc[1]
                    else:
                        distance = pqz[0] - pqz[1]
                    #print(distance)
                    #print(pqz)
                    #print(pqd)

                    #print(distance)
                    if ( (abs(distance)) > (pqd[1] * Rtol)):
                        L[i] = 1
                        #print("ow")
                    
                    #print(pqd)

                    if ( (np.sqrt( pqd[1] ** 2+distance ** 2 ) / RA) > (Atol) ):
                        L[i] = 1


                #print(L)
                dE[j-1] = sum(L) / n

                #print(L)
                print(sum(L))
                #print(n)
                print(dE)

                if speed == 1:
                    if (dE[j] == 0):
                        dim = j

                    if j >= 3 and ((dE[j-2] > dE[j-1] and (dE[j-1] < dE[j]))):
                        dim = j-1

                    if j >= 2 and abs(dE[j] - dE[j-1]) <= 0.001:
                        dim = j
            if speed == 0:
                i = 2
                for i in range(2, len(dE) - 1):
                    if (dE[i] == 0) or ((dE[i-1] > dE[i] or (dE[i] < dE[i+1]))):
                        dim = i

                    if dE[i-1] - dE[i] <= 0.001:
                        dim = i-1
            print(dE)
            

            if not dim:
                dim = MaxDim
                print("No dimenstion found, dim set to MaxDim\n")

        def kd_part(y_in, z_in, bin_size):

            # Global y_model, z_model, sort_list, and node_list
            global y_model
            global z_model
            global node_list
            global sort_list

            y_model = y_in
            #print(y_model)
            z_model = z_in

            # d: dimension of phase space
            # n_y: number of points to put into partitioned database
           # for i in range(0, len(y_model)):
                #print(y_model[i])

            #print(np.shape(y_model))
            [n_y, d] = y_model.shape

            # set up first node
            node_list = [1, n_y, 0, 0]
            sort_list = [0, 0]

            # node info?
            node = 1
            last = 1

            while node <= last:
                segment = range(node_list[1],node_list[2])
                segment = pd.DataFrame(segment)

                i = list(range(1, d))
                maxList = np.amax(y_model, axis = 1)
                minList = np.amin(y_model, axis = 1)
                node_range = maxList - minList

                if max(node_range) > 0 and len(segment) >= bin_size:
                    [temp, index] = np.sort(node_range)
                    temp = None
                    yt = y_model[segment, :]
                    zt = z_model[segment, :]
                    y_sort = yt.sort(axis=1)
                    y_index = yt.argsort(axis=1)
                    # [y_sort,y_index] = sort(yt[:,index[d]])   ??

                    [tlen, temp] = np.size(yt)
                    temp = None

                    tlen = 6  # remove
                    if math.remainder(tlen, 2):
                        cut = y_sort([tlen+1]/2)
                    else:
                        cut = (y_sort[tlen/2]+y_sort[tlen/2 + 1])/2

                    L = y_sort <= cut

                    if sum(L) == tlen:
                        L = y_sort < cut
                        cut = (cut+max(y_sort[L]))/2

                    y_model[segment, :] = yt[y_index, :]
                    z_model[segment, :] = zt[y_index, :]

                    sort_list[node, :] = [index(d), cut]
                    sort_list[node, 3] = last + 1
                    sort_list[node, 4] = last + 2
                    last = last + 2

                    # edit this!
                    node_list = np.r_[node_list, segment(
                        1), segment(1)+sum(L)-1, 0, 0]
                    node_list = np.r_[node_list, segment(
                        1)+sum(L), segment(tlen), 0, 0]
                    sort_list = np.r_[sort_list, 0, 0, 0, 0]

                node = node + 1

        def kdSearch(node, m_search, yq, y_model, z_model, L_done, b_upper, b_lower, sort_list, node_list):
            global pqd
            global pqr
            global pqz

            if L_done:
                return

            #print(node_list)
            np.asarray(node_list)

            if node == 1:
                nodeCheck = node_list[2]
                nodeDistance = node_list[0:2]
                #print(nodeDistance)
            else:
                nodeCheck = node_list[node][2]
                nodeDistance = node_list[node] [0:2]

            if nodeCheck == 0:

                yi = nodeDistance
                #print(y_model)
                #print(yi)
                if len(yi) == 2:
                    
                    yt = y_model[(yi[0]-1):yi[1]]
                    zt = z_model[(yi[0]-1):yi[1]]
                else:
                    yt = y_model[yi[0]:yi[1], :]
                    zt = z_model[yi[0]:yi[1], :]

                d = len(yq)
                #print(d)

                j = list(range(1, d+1))
                #print(j)
                #print(yt)
                #print(yq)

                yt = pd.DataFrame(yt)
                zt = pd.DataFrame(zt)

                tempSum = 0
                newSum = 0

                # subSeries = yq.squeeze()

                # for i in range(0, d):
                #     templist = yt.iloc[:,i]
                #     templist = list(map(lambda x: abs(x - yq[i]), templist))
                #     tempList = yt.iloc[:,i].subtract(subSeries, fill_value = 0, axis = 0)

                #print("gabagool 1")
                #print(yq)
                yqArray = yq.to_numpy()
                #print(yqArray)
                #yqArray = yq.values.tolist()
                #print(yqArray)
                #print("gabagool 2")
                addToLoc = []
                column_list = list(yq)
                for h in column_list:
                    #print(yq[j])
                    addToLoc.append(int(yq[h]))

                


                tempArray = yt.to_numpy()
                [rows, cols] = tempArray.shape
                if cols == 1:
                    for i in range(0, len(j)):
                        #print(yqArray[i])
                        templist = yt.iloc[:,i]
                        templist = list(map(lambda x: abs(x - np.array(yqArray[i])), templist))
                        powerList = np.power(templist,2)
                        powerList = np.sqrt(powerList)
                        tempSum += powerList
                else:
                    for i in range(0, len(j)+1):

                        templist = yt.iloc[:,i]
                        #print(templist)
                        templist = list(map(lambda x: abs(x - np.array(addToLoc[i])), templist))
                        powerList = np.power(templist,2)
                        powerList = np.sqrt(powerList)
                        tempSum += powerList

                

                # subSeries = yq.squeeze()
                # tempArray = yt.to_numpy
                # subList = yt.subtract(subSeries, fill_value = 0, axis = 0)
                # powerList = np.power(subList,2)
                # powerList = np.sqrt(powerList)
                # newSum += powerList


                # subSeries = yq.squeeze()
                # tempArray = yt.to_numpy
                # subList = list(map(lambda x: abs(x - subSeries), tempArray))
                # powerList = np.power(subList,2)
                # powerList = np.sqrt(powerList)
                # newSum += powerList
                    

                dist = tempSum
                #print(dist)

                dist = np.insert(dist,len(dist), 100)
                dist = np.insert(dist,len(dist), 101)
                #dist = dist.tolist()
                #print(dist)
                #dist = np.array(dist)
                index = dist.argsort(axis = 0, kind = 'mergesort')
                #print(index) 
                #print(dist)

                dist.sort(axis = 0)
                
                if pqd:
                    pqd = pqd,dist
                    #print(pqd)
                    print("hello")
                else:
                    pqd = dist
                    #print("rar")
                
                #print(pqd)

                pqr_frame = pd.DataFrame(yt)
                for column in pqr:
                    pqr_frame = pqr_frame.assign(newColumn = pqr[column])
                pqr = pqr_frame


                pqz_frame = pd.DataFrame(zt)
                for column in pqz:
                    pqz_frame = pqz_frame.assign(newColumn = pqz[column])
                pqz = pqz_frame

                

                [length, temp] = pqz.shape
                #print(pqz)

                #print(pqr)
                #print(zt)
                #print(pqz)
                #print(len(pqz))
                #print(len(index))
                index = [x+1 for x in index]

                indexList = []
                pqrlist = []
                pqzlist = []
                #print(pqr)
                

                
                tempconvert = pqr.values.tolist()
                removeTuple = []
                frameListR = []
                frameListZ = []

                if len(pqr.axes[1]) > 1:

                    for i in range(0, len(pqr.axes[1])):
                        tempColumn = pqr.iloc[:,i]
                        frameListR.append(tempColumn.tolist())
                    for i in range(0, len(pqz.axes[1])):
                        tempColumn = pqz.iloc[:,i]
                        frameListZ.append(tempColumn.tolist())

                else:
                    for i in range(0, len(pqr)):
                        pqrlist.append(int(pqr.iloc[i]))
                    for i in range(0, len(pqz)):
                        pqzlist.append(int(pqz.iloc[i]))

                tempListPqr = []
                tempListPqz = []

                for i in range(0, len(index)):
                    #print(index)
                    indexList.append(index[i] - 1)

                if len(index) > length:
                    tempList1 = []
                    tempList2 = []

                    #print(len(frameListR))
                    #print(len(indexList))
                    #print(frameListR[0][1])
                    counter = 0
                    if len(pqr.axes[1]) > 1:
                        for i in range(0, len(frameListR)):
                            for j in range(0,len(frameListR[0])):
                                tempList1.append(frameListR[i][indexList[j]])
                                counter+=1
                                print(tempList1)
                            tempListPqr.append(tempList1)
                        pqr = tempListPqr
                        
                        
                        for i in range(0, len(frameListZ)):
                            for j in range(0, len(frameListZ[0])):
                                tempList2.append(frameListZ[i][indexList[j]])
                            tempListPqz.append(tempList2)
                        pqz = tempListPqz

                        flat_pqr_list = []
                        for sublist in pqz:
                            for item in sublist:
                                flat_pqr_list.append(item)
                        pqz = flat_pqr_list

                    else:
                    
                        for j in range(0,len(pqr)):
                            tempListPqr.append(pqrlist[indexList[j]])
                            #print(tempListPqr)
                        pqr = tempListPqr
                        #print(pqr)
                        

                        for j in range(0,len(pqz)):
                            tempListPqz.append(pqzlist[indexList[j]])
                        pqz = tempListPqz

                        #pqr = pqr.iloc[index[0:len(pqr)]]
                        #pqz = pqz.iloc[index[0:len(pqz)]]

                        #print(indexList)
                        #print(pqr)
                        #print(pqz)
                        
                else:
                    pqr = pqr.iloc[index]
                    pqz = pqz.iloc[index]
                    print("hello")

                    for j in range(0,len(pqr)):
                        tempList1.append(pqrlist[indexList[j]])

                    for j in range(0,len(pqz)):
                        tempList2.append(pqzlist[indexList[j]])


                if len(pqd) > m_search:
                    pqd = pqd[0:m_search]

                #print(pqd)


                #[length, temp] = pqz.shape
                if len(pqz) > m_search:
                    pqr = pqr[0:m_search]
                    pqz = pqz[0:m_search]
                    
                    #print(pqr)
                    #print(pqz)
            

                
                #print()

                #print(pqd)
                #print(m_search)
                #print(pqd[1])
                #print(yq-b_lower)
                #print(yq-b_upper)
                #print(pqd[m_search-1])

                #x = kdSearch()

                if any( ((np.abs(yq-b_lower)) <= (pqd[m_search-1])) | ((np.abs(yq-b_upper)) <= (pqd[m_search-1])) ):
                    L_done = 1
                #print(pqz)
                #print("why")
                
            
            else:
                print("hello")
                disc = sort_list(node, 1)
                part = sort_list(node, 2)
            
            

                if yq[disc] <= part:
                    temp = b_upper[disc]
                    b_upper[disc] = part
                    kdSearch(node_list(node, 3), m_search, yq, y_model, z_model, L_done, b_upper, b_lower, sort_list, node_list)

                    b_upper[disc] = temp
                else:
                    temp = b_lower[disc]
                    b_lower[disc] = part

                    kdSearch(node_list(node, 4), m_search, yq, y_model, z_model, L_done, b_upper, b_lower, sort_list, node_list)

                    b_lower[disc] = temp

                if L_done:
                    return
                if yq[disc] <= part:
                    temp = b_lower[disc]
                    b_lower[disc] = part

                    L = getOverlap(yq, m_search, pqd, b_upper, b_lower)

                    if L:
                        kdSearch(node_list(node, 4), m_search, yq, y_model, z_model, L_done, pqr, pqz, b_upper, b_lower, sort_list, node_list)

                    b_lower[disc] = temp

                else:
                    temp = b_upper[disc]
                    b_upper[disc] = part
                    L = getOverlap(yq, m_search, pqd, b_upper, b_lower)

                    if L:
                        kdSearch(node_list(node, 3), m_search, yq, y_model, z_model, L_done, b_upper, b_lower, sort_list, node_list)

                    b_upper[disc] = temp
                
                if L_done:
                    return

            def getOverlap(yq, m_search, pqd, b_upper, b_lower):
                dist = pqd[m_search] ** 2
                sum = 0
                i = 1
                for i in len(yq):

                    if yq[i] < b_lower[i]:
                        sum = sum + (yq[i] - b_lower[i]) ** 2
                        if sum > dist:
                            L = 0
                            return L
                    elif yq[i] > b_upper[i]:
                        sum = sum + (yq[i] - b_upper[i]) ** 2
                        if sum > dist:
                            L = 0
                            return L
                L = 1
                return L
                # return max(0, min(a[1], b[1], c[1], d[1], e[1]) - max(a[0], b[0], c[0], d[0], e[0]))
        FNN2020(data, tau, MaxDim, Rtol, Atol, speed)
    if __name__ == "__main__":
        main()
