"""
PARTE 1. Convergencia de sumas inferior y superior.

1. Implemente en python una función que tome la cantidad de puntos de la
partición (equispaciada) en [-1,1] y calcule la suma inferior de la función.
Repita para la suma superior.

2. Genere tablas comparativas de ambas sumas variando el tamaño de la partición.
En esta tabla debe incluir los resultados de la aproximación y el residuo (la
diferencia entre la aproximación y el valor de π) para cada suma.
Una tabla debe variar el tamaño de N de 10 a 100 variando de a 10, la segunda
de 100 a 1000 variando de a 100 y la tercera de 1000 a 10000 variando de a 1000.

Convención:
El parámetro N representa la cantidad de PUNTOS de la partición P.

P = {x_0(=a), x_1, x_2, ..., x_n(=b)}
N = #P

Genera N-1 subintervalos indexados desde i = 0 hasta i = N-2,
donde el subintervalo i es [x_i, x_{i+1}].
"""


import numpy as np
import csv


def f(x):
    """f(x) = 2*sqrt(1 - x^2)"""
    return 2.0 * np.sqrt(1.0 - x**2)


def lower_sum(N):
    """
    Suma inferior de Riemann de f en [-1, 1] con partición equispaciada
    de N puntos.
    """
    if N < 2:
        raise ValueError("Se requieren al menos 2 puntos de partición.")

    # Genera N números reales igualmente espaciados entre [-1,1].
    x = np.linspace(-1.0, 1.0, N)
    
    # Base de los rectángulos de aproximación.
    delta = x[1] - x[0]
    
    # Evalúa la función f en todos los puntos de la partición y guarda los resultados.
    f_vals = f(x)

    # Construye un arreglo donde la i-ésima entrada es el mínimo entre f(x_i) y f(x_i+1).
    m = np.minimum(f_vals[:-1], f_vals[1:])
    
    # Hace la suma inferior.
    return delta * np.sum(m)


def upper_sum(N):
    """
    Suma superior de Riemann de f en [-1, 1] con partición equispaciada
    de N puntos.
    """
    if N < 2:
        raise ValueError("Se requieren al menos 2 puntos de partición.")

    x = np.linspace(-1.0, 1.0, N)
    delta = x[1] - x[0]
    f_vals = f(x)

    M = np.maximum(f_vals[:-1], f_vals[1:])

    return delta * np.sum(M)


def generate_table(N_values, file_name):
    """
    Genera una tabla con columnas:
    N,
    suma_inferior,
    suma_superior,
    pi,
    residuo_inferior,
    residuo_superior.
    Guarda en CSV y también imprime por consola.
    """
    headers = [
        "N",
        "suma_inferior",
        "suma_superior",
        "pi",
        "residuo_inferior",
        "residuo_superior"]
 
    rows = []
    for N in N_values:
        L = lower_sum(N)
        U = upper_sum(N)
        pi = np.pi
        remainder_L = L - np.pi
        remainder_U = U - np.pi
        rows.append([N, L, U, pi, remainder_L, remainder_U])
 
    # Guardar CSV.
    with open(file_name, "w", newline="") as file:
        my_writer = csv.writer(file, delimiter=";")
        my_writer.writerow(headers)
        for row in rows:
            N = row[0]
            # N como entero, los demás con suficientes decimales.
            # Cambia el . por , para poder graficar en Excel.
            formated_values = [f"{v:.10f}".replace(".", ",") for v in row[1:]]
            my_writer.writerow([N] + formated_values)
 
    # Imprimir por consola.
    print(f"{'N':>6} | {'suma inferior':>20} | {'suma superior':>20} | {'pi':>20} | {'suma inferior - pi':>20} | {'suma superior - pi':>20}")
    print("-" * 125)
    for row in rows:
        N, L, U, pi, remainder_L, remainder_U = row
        print(f"{N:>6} | {L:>20.10f} | {U:>20.10f} | {pi:>20.10f} | {remainder_L:>20.10f} | {remainder_U:>20.10f}")


# Una única tabla: combina los tres rangos (10, 100, 1000).
N_values = sorted(set(
    list(range(10, 101, 10)) +
    list(range(200, 1001, 100)) +
    list(range(2000, 10001, 1000))
))

generate_table(N_values, "parte_1/tabla_convergencia_sumas.csv")