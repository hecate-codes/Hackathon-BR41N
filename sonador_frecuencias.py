import numpy as np
import sounddevice as sd
import pandas as pd
import ast

with open('musica.txt', 'r', encoding='utf-8') as f:
    # Nos saltamos las 3 primeras líneas que son el título y separador
    lineas = f.readlines()[3:]

datos_limpios = []

for linea in lineas:
    if linea.strip(): # Ignoramos líneas en blanco si las hay
        # Separamos el número de lista ("1. ") de la lista real ("[ [261.63]... ]")
        solo_lista_str = linea.split('. ', 1)[1].strip()
        
        # ast.literal_eval convierte el texto de la lista en una lista real de Python de forma segura
        elemento = ast.literal_eval(solo_lista_str)
        
        # Guardamos los datos en un diccionario
        datos_limpios.append({
            "Frecuencias (Hz)": elemento[0],
            "Nota": elemento[1],
            "Duración (ms)": elemento[2]
        })

df = pd.DataFrame(datos_limpios)
matriz_numpy = df.to_numpy()

for i in range(matriz_numpy.shape[0]):
    # matriz_numpy tienes tres columnas, la primera es la frecuencia (que está dentro de otra lista), la segunda es la nota asociada y la última la duración.
    frecuencia = matriz_numpy[i][0][0]
    duracion = matriz_numpy[i][2] *10**-3     # Segundos
    frecuencia_muestreo = 44100  # Calidad estándar de CD (Hz)

    # Generar los puntos de tiempo
    t = np.linspace(0, duracion, int(frecuencia_muestreo * duracion), False)

    # Generar la onda senoidal
    onda = 0.5 * np.sin(2 * np.pi * frecuencia * t)
    sd.play(onda, frecuencia_muestreo)
    sd.wait()
