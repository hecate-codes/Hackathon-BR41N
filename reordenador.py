import numpy as np
import pandas as pd
import ast

data_test = np.load("datos_procesados_triple/P2_post_test.mat.npz")
X = data_test['X']
Y = data_test['y']
data_ordenadoX = np.zeros_like(X)
data_ordenadoY = np.zeros_like(Y)

# 1. Obtenemos las posiciones (índices) de cada clase
idx_12 = np.where((Y == 1) | (Y == 2))[0] # Donde es 1 o 2
idx_0  = np.where(Y == 0)[0]               # Donde es 0

# 2. Vemos cuántas parejas podemos formar
# (Necesitamos que haya la misma cantidad de ambos para alternar perfectamente)
n_parejas = min(len(idx_12), len(idx_0))

# 3. Creamos un array de índices intercalados: [idx_12[0], idx_0[0], idx_12[1], idx_0[1]...]
indices_intercalados = np.empty(2 * n_parejas, dtype=int)
indices_intercalados[1::2] = idx_12[:n_parejas] # Posiciones pares
indices_intercalados[0::2] = idx_0[:n_parejas]  # Posiciones impares

# 4. Creamos las matrices finales usando esos índices
data_ordenadoX = X[indices_intercalados]
data_ordenadoY = Y[indices_intercalados]

np.savez('datos_ordenados.npz', X=data_ordenadoX, y=data_ordenadoY)

    

