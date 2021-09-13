# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:35:13 2021

@author: Pc Planet
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

temp_in = [0.72,0.61,0.65,0.68,0.75,0.90,1.02,0.93,0.85,0.99,1.02]

plt.plot(years,temp_in)
plt.xlabel('Years')
plt.ylabel('Temp_Index')
plt.title('Global Warming',{'fontsize':12,'horizontalalignment':'center'})
plt.show()

# line graph

Months = ['Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec']

customer1 = [12,13,9,8,7,8,8,7,6,5,8,10]
customer2 = [14,16,11,7,6,6,7,6,5,8,9,12]

plt.plot(Months,customer1,color = "red",label = "Customer1",marker = "o")
plt.plot(Months,customer2,color = "blue",label = "Customer2",marker = "^")
plt.xlabel('Months')
plt.ylabel("Electricity consumption")
plt.title("Consumption/Month",{'fontsize':12,'horizontalalignment':'center'})
plt.legend()
plt.show()

# separating the graphs
plt.subplot(1,2,1)
plt.plot(Months,customer1,color = "red",label = "Customer1",marker = "o")
plt.xlabel('Months')
plt.ylabel("Electricity consumption")
plt.title("Consumption/Month(Customer1)",{'fontsize':12,'horizontalalignment':'center'})
plt.show()

plt.subplot(1,2,2)
plt.plot(Months,customer2,color = "blue",label = "Customer2",marker = "^")
plt.xlabel('Months')
plt.ylabel("Electricity consumption")
plt.title("Consumption/Month(Customer2)",{'fontsize':12,'horizontalalignment':'center'})
plt.show()

# plotting scatter
plt.scatter(Months,customer1,color = "red",label = "Customer1")
plt.scatter(Months,customer2,color = "blue",label = "Customer2")
plt.xlabel("Months")
plt.ylabel("Electricity Consumption")
plt.title("Scatter plotting for Consumption/Months")
plt.grid()
plt.legend()
plt.show()

# plotting histogram
plt.hist(customer1,bins = 15,color = "orange")
plt.xlabel("Months")
plt.ylabel("Electricity consumption")
plt.title("Histogram")
plt.show()

# plottin bar-chart
plt.bar(Months,customer1,width = 0.8,color ="red")
plt.xlabel("Months")
plt.ylabel("Electricity consumption")
plt.title("Bar-chart")
plt.show()

customer3 = [2,4,6,8,10,12,3,7,9,11,13,15]
bar_wid = 0.2
Month_np = np.arange(12)

plt.bar(Month_np,customer1,bar_wid,color = "red",label = "Customer1")
plt.bar(Month_np+bar_wid,customer2,bar_wid,color = "blue",label = "Customer2")
plt.bar(Month_np+bar_wid*2,customer3,bar_wid,color = "brown",label = "Customer3")
plt.xlabel("Months")
plt.ylabel("Electricity consumption")
plt.title("Bar-chart for Consumption/Months")
plt.legend()
plt.show()
plt.xticks(Month_np+(bar_wid)/12,
           ('Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec'))


# box plot
plt.boxplot(customer1, notch= False, vert= True)

plt.boxplot([customer1,customer2],patch_artist=True,
            boxprops=dict(facecolor='red',color='red'),
            whiskerprops=dict(color='green'),
            capprops=dict(color='blue'),
            medianprops=dict(color='yellow'))
plt.show()










