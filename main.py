import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing.pool import Pool

# Función para rotar la imagen de manera secuencial


def rotate_image_sequential(image):
    rows, cols = image.shape
    new_image = np.zeros((cols, rows), dtype=np.uint8)  # Imagen rotada
    for i in range(rows):
        for j in range(cols):
            new_image[j, rows - i - 1] = image[i, j]
    return new_image

# Función para rotar la imagen de manera paralela


def rotate_image_parallel(image, num_processes):
    rows, cols = image.shape
    # 🚩PARTICIÓN: Dividir la imagen en partes iguales según el número de procesos.
    rows_per_process = rows // num_processes

    # 🚩COMUNICACIÓN: Crear un pool de procesos para manejar la ejecución paralela.
    with Pool(processes=num_processes) as pool:

        # 🚩MAPPING: Asignar cada parte de la imagen a un proceso del pool.
        result_parts = pool.map(rotate_image_sequential,
                                [image[i * rows_per_process:(i + 1) * rows_per_process]
                                 for i in range(num_processes)])  # Solo pasamos la parte de la imagen

    new_image = np.zeros((cols, rows), dtype=np.uint8)

    # 🚩AGLOMERACIÓN: Combinar los resultados de cada proceso en una sola imagen rotada.
    for i in range(num_processes):
        new_image[:, i * rows_per_process:(i + 1) *
                  rows_per_process] = result_parts[num_processes - i - 1]
    return new_image


###############################################################################
#                           FUNCIONES AUXILIARES                               #
###############################################################################


def throughput(tasks, total_time):
    """
    Calcula el Throughput (tasa de procesamiento).
    T = número de tareas / tiempo total
    """
    return tasks / total_time


def latency(total_time, tasks):
    """
    Calcula la Latencia (tiempo promedio por tarea).
    L = tiempo total / número de tareas
    """
    return total_time / tasks


def efficiency(speedup, num_cores):
    """
    Calcula la eficiencia de la ejecución paralela.
    E = S / N
    """
    return speedup / num_cores


def speedup(serial_time, parallel_time):
    """
    Calcula el Speedup de la ejecución paralela.
    S = T_s / T_p
    """
    return serial_time / parallel_time


def overall_speedup(serial_time, parallel_time, num_cores):
    """
    Calcula el Speedup general teniendo en cuenta la Ley de Amdahl.
    S = 1 / ((1 - P) + (P / N))
    donde P es la fracción paralelizable y N es el número de núcleos.
    """
    p = (serial_time - parallel_time) / serial_time
    return 1 / ((1 - p) + (p / num_cores))


if __name__ == '__main__':
    # Cargar la imagen en escala de grises
    img = Image.open("imagen.jpg").convert('L')  # Convertir a escala de grises
    img_array = np.array(img)

    # Medir el tiempo de la rotación secuencial
    start_time = time.time()
    rotated_image_sequential = rotate_image_sequential(img_array)
    end_time = time.time()
    sequential_time = end_time - start_time

    # Medir el tiempo de la rotación paralela
    num_processes = 8  # Número de núcleos
    start_time = time.time()
    rotated_image_parallel = rotate_image_parallel(img_array, num_processes)
    end_time = time.time()
    parallel_time = end_time - start_time

    # Mostrar imágenes
    plt.figure(figsize=(10, 5))

    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title("Imagen Original")
    plt.axis("off")

    # Imagen rotada secuencialmente
    plt.subplot(1, 3, 2)
    plt.imshow(rotated_image_sequential, cmap='gray')
    plt.title(f"Rotación Secuencial\nTiempo: {sequential_time:.4f} s")
    plt.axis("off")

    # Imagen rotada paralelamente
    plt.subplot(1, 3, 3)
    plt.imshow(rotated_image_parallel, cmap='gray')
    plt.title(f"Rotación Paralela\nTiempo: {parallel_time:.4f} s")
    plt.axis("off")

    # Mostrar el gráfico con las imágenes
    plt.show()

    # Imprimir los tiempos
    print(f"Tiempo de ejecución secuencial: {sequential_time:.4f} segundos")
    print(f"Tiempo de ejecución paralelo: {parallel_time:.4f} segundos")

    ###############################################################################
    #                           CÁLCULO DE MÉTRICAS                                #
    ###############################################################################

    # Número de tareas (en este caso, igual al número de procesos)
    tasks = num_processes
    # Número de núcleos
    num_cores = num_processes

    # Calcular métricas
    T = throughput(tasks, parallel_time)
    L = latency(parallel_time, tasks)
    S = speedup(sequential_time, parallel_time)
    E = efficiency(S, num_cores)
    OS = overall_speedup(sequential_time, parallel_time, num_cores)

    # Imprimir métricas
    print(f"Throughput: {T:.2f} tasks/sec")
    print(f"Latency: {L:.4f} sec/task")
    print(f"Speedup: {S:.2f}")
    print(f"Efficiency: {E:.2f}")
    print(f"Overall Speedup: {OS:.2f}")
