%% Cargar archivo
clear; clc;
data = load('P1_pre_training.mat');

%% Extraer variables
y = data.y;        % señal EEG (muestras x canales)
fs = data.fs;      % frecuencia de muestreo
trig = data.trig;  % triggers

%% Información básica
[n_muestras, n_canales] = size(y);

fprintf('--- INFORMACIÓN EEG ---\n');
fprintf('Número de muestras: %d\n', n_muestras);
fprintf('Número de canales: %d\n', n_canales);
fprintf('Frecuencia de muestreo: %.2f Hz\n', fs);

%% Duración total
duracion = n_muestras / fs;
fprintf('Duración total: %.2f segundos (%.2f minutos)\n', duracion, duracion/60);

%% Información de triggers
fprintf('\n--- TRIGGERS ---\n');

% Ver valores únicos de triggers (tipos de evento)
tipos_trig = unique(trig);
disp('Tipos de triggers:');
disp(tipos_trig');

% Encontrar posiciones donde hay triggers (distintos de 0 normalmente)
idx_trig = find(trig ~= 0);

fprintf('Número total de eventos: %d\n', length(idx_trig));

% Mostrar primeros eventos
n_mostrar = min(10, length(idx_trig));

fprintf('\nPrimeros %d eventos:\n', n_mostrar);
for i = 1:n_mostrar
    fprintf('Evento %d: tipo=%d, muestra=%d, tiempo=%.3f s\n', ...
        i, ...
        trig(idx_trig(i)), ...
        idx_trig(i), ...
        idx_trig(i)/fs);
end

%% Visualizar señal (canal 1)
figure;
plot((1:1000)/fs, y(1:1000,1));
xlabel('Tiempo (s)');
ylabel('Amplitud');
title('EEG Canal 1 (primeros 1000 samples)');
grid on;






%% Visualizar triggers sobre señal (opcional)
figure;
plot((65839:67839)/fs, y(65839:67839,1)); hold on;

% marcar triggers en ese rango
%%idx_trig_plot = idx_trig(idx_trig <= 1000);
%%stem(idx_trig_plot/fs, y(idx_trig_plot,1), 'r');

xlabel('Tiempo (s)');
title('EEG + Triggers');
legend('Señal','Triggers');
grid on;

%% Detectar inicios de tareas
trig_bin = trig ~= 0;  % convertir a 0/1 (activo o no)

% detectar cambios
cambios = diff([0; trig_bin]);  

% inicios (0 -> 1)
idx_inicio = find(cambios == 1);

% finales (1 -> 0)
idx_fin = find(cambios == -1);

n_tareas = length(idx_inicio);

fprintf('\n--- TAREAS DETECTADAS ---\n');
fprintf('Número de tareas: %d\n', n_tareas);

%% Mostrar primeras tareas
n_mostrar = min(10, n_tareas);

for i = 1:n_mostrar
    duracion = (idx_fin(i) - idx_inicio(i)) / fs;
    
    fprintf('Tarea %d: inicio=%.2f s, fin=%.2f s, duracion=%.2f s\n', ...
        i, ...
        idx_inicio(i)/fs, ...
        idx_fin(i)/fs, ...
        duracion);
end