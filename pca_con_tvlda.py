import torch 
import numpy as np 
import scipy.io as sio
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tvlda import TVLDA

def filtro_paso_banda(data, lowcut, highcut, fs, order=4):
    """Aplico un PASO BANDA"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Filtramos a lo largo del eje del tiempo (axis=0)
    y_filtrado = filtfilt(b, a, data, axis=0)
    return y_filtrado

def preparar_datos_bci(ruta_archivo, tam_ventana_segundos=0.5, aplicar_filtro=True):
    mat = sio.loadmat(ruta_archivo)
    fs = mat['fs'][0, 0]
    y_raw = mat['y']
    trig = mat['trig'].flatten()
    
    # 1. Filtro paso banda (Opcional pero altamente recomendado)
    if aplicar_filtro:
        y = filtro_paso_banda(y_raw, lowcut=8.0, highcut=30.0, fs=fs)
    else:
        y = y_raw
        
    ensayos_X = []
    etiquetas_Y = []
    
    offset_start = 2 * fs                   # 2 segundos de descarte
    useful_duration = 6*fs          # 6 segundos de señal útil
    window_duration = int(tam_ventana_segundos * fs)         # Ventanas de 1.5 segundos
    n_windows_per_trial = useful_duration // window_duration 
    
    for clase in [1, -1]:
        # Búsqueda vectorizada rápida de bloques (Equivalente a tu bucle while)
        es_clase = (trig == clase).astype(int)
        cambios = np.diff(np.concatenate(([0], es_clase, [0])))
        inicios = np.where(cambios == 1)[0]
        fines = np.where(cambios == -1)[0]
        
        for ini, fin in zip(inicios, fines):
            # Nos aseguramos de que el bloque grabado tiene al menos 8 segundos
            if (fin - ini) >= (offset_start + useful_duration):
                
                start_sample = ini + offset_start
                end_sample = start_sample + useful_duration
                
                # Recortamos solo la porción útil (6 segundos)
                ensayo = y[start_sample:end_sample, :]
                
                ensayo_ventanas = ensayo.reshape(n_windows_per_trial, window_duration, -1)
                
                varianza = np.var(ensayo_ventanas, axis=1) # Forma resultante: (4, 16)
                caracteristicas = np.log(varianza)
                
                caracteristicas = caracteristicas.T # Forma final: (16, 4)
                
                ensayos_X.append(caracteristicas)
                etiquetas_Y.append(clase) 

    # Convertir a tensores de PyTorch
    X_tensor = torch.tensor(np.array(ensayos_X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(etiquetas_Y), dtype=torch.float32)
    
    return X_tensor, y_tensor
   

def pipeline_pca_tvlda(X_train, y_train, X_test, n_componentes=8):
    """Aplicamos PCA primero"""
    N_train, F_canales, W_ventanas = X_train.shape
    N_test, _, _ = X_test.shape

    # Aplanar porque el PCA funciona en 2D y el TVLDA en 3D
    X_train_2d = X_train.cpu().numpy().transpose(0,2,1).reshape(-1, F_canales)
    X_test_2d = X_test.cpu().numpy().transpose(0,2,1).reshape(-1, F_canales)

    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)

    pca = PCA(n_components=n_componentes)
    X_train_pca_2d = pca.fit_transform(X_train_2d_scaled)
    X_test_pca_2d = pca.transform(X_test_2d_scaled)

    # Ahora volvemos al 3D para el TVLDA
    X_train_pca_3d = X_train_pca_2d.reshape(N_train, W_ventanas, n_componentes).transpose(0, 2, 1)
    X_test_pca_3d  = X_test_pca_2d.reshape(N_test, W_ventanas, n_componentes).transpose(0, 2, 1)

    X_train_final = torch.tensor(X_train_pca_3d, dtype=torch.float32)
    X_test_final = torch.tensor(X_test_pca_3d, dtype=torch.float32)

    clasificador = TVLDA(lamb=1e-4, device="cpu")
    clasificador.fit(X_train_final, y_train, label_xa=1)
    predicciones = clasificador.predict(X_test_final)

    return predicciones, pca, clasificador



if __name__ == "__main__":
    X_train, y_train = preparar_datos_bci('P1_post_training.mat', tam_ventana_segundos=0.5)

    X_test, y_test = preparar_datos_bci('P1_post_test.mat', tam_ventana_segundos=0.5)
    
    print(f"\nForma del Tensor de Entrenamiento: {X_train.shape}")
    print(f"Forma del Tensor de Test: {X_test.shape}")
    
    # Reducimos los 16 canales originales a 8 componentes principales
    predicciones, pca_model, tvlda_model = pipeline_pca_tvlda(X_train, y_train, X_test, n_componentes=8)
    
    # 4. Calcular Métricas de Precisión
    correctas = (predicciones == y_test).sum().item()
    total_ensayos = len(y_test)
    precision = (correctas / total_ensayos) * 100
    
    print(f"Accuracy del modelo:   {precision:.2f}%")