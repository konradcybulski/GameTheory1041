import numpy as np

population = np.array([0, 1, 2, 3, 1])
reputation = np.array([0, 1, 0, 1, 1])
y_array = np.array([0, 1, 2, 3, 4])

arr_len = len(y_array)

Pstrat = population[y_array]
Prep = reputation[y_array]

XStrategy = [0, 1]
XActionBad = XStrategy[0]
XActionGood = XStrategy[1]
Cx = np.zeros(arr_len, dtype=int)
Cy = np.zeros(arr_len, dtype=int)


Cx_temp = (XActionGood*(1 - Prep) + XActionBad*Prep)
coop_vector = Cx_temp == np.ones(arr_len)
if np.random.random() < 1:
    elements_to_change = int(arr_len * 0.5)
    mask = np.random.randint(arr_len,size=elements_to_change)
    Cx_temp[mask] = 1 - Cx_temp[mask]
Cx = Cx_temp