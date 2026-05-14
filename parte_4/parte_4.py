"""
SECCIÓN 4. Integración Montecarlo

Los métodos de integración del punto anterior son modificaciones directas de
la teoría inicial. Aún así, existen muchos otros algoritmos para resolver
este problema. Tomemos ahora un enfoque probabilístico.

1. Investigue qué es el método Montecarlo.

2. Calcule el valor de la integral de f utilizando 2 generadores de números
aleatorios entre 0 y 1 para distintas cantidades de puntos, siguiendo la misma
lógica que los ejercicios anteriores.

3. Grafique los resultados obtenidos para el valor de π por este método.

Método utilizado:
  - Se dibuja un rectángulo R = [-1, 1] x [0, 2] que contiene la región bajo
    la curva f(x) = 2*sqrt(1 - x^2).
  - El rectángulo tiene área 4 y la región bajo la curva tiene área pi.
  - Se sortean N puntos uniformes (x, y) en el rectángulo y se cuenta cuántos
    caen bajo la curva (hits).
  - Por la ley de los grandes números: hits/N converge a pi/4, por lo tanto
    pi ≈ 4 * hits / N.

Convención:
A diferencia de las secciones anteriores, N no representa puntos de una
partición sino la cantidad de muestras aleatorias del experimento.
"""


import numpy as np
import csv


def f(x):
    """f(x) = 2*sqrt(1 - x^2)"""
    return 2.0 * np.sqrt(1.0 - x**2)


# Dos generadores aleatorios independientes para las coordenadas x e y
# (mismo resultado en cada ejecución).
rng_x = np.random.default_rng(seed=42)
rng_y = np.random.default_rng(seed=2024)


def montecarlo_pi(N):
    """
    Estimación de pi mediante el método de Montecarlo con N
    muestras. Sortea N puntos uniformes en el rectángulo [-1, 1] x [0, 2] y
    cuenta cuántos caen bajo la curva f(x) = 2*sqrt(1 - x^2).
    """
    # Genera N coordenadas (u, v) uniformes en [0, 1] usando dos generadores
    # independientes.
    u = rng_x.uniform(0.0, 1.0, size=N)
    v = rng_y.uniform(0.0, 1.0, size=N)

    # Mapea al rectángulo [-1, 1] x [0, 2].
    x = 2.0 * u - 1.0
    y = 2.0 * v

    # Cuenta cuántos puntos cumplen y <= f(x), es decir, caen bajo la curva.
    hits = np.sum(y <= f(x))

    # Estima pi como 4 * (proporción de puntos bajo la curva).
    return 4.0 * hits / N


def generate_table(N_values, file_name):
    """
    Genera una tabla con columnas:
    N,
    montecarlo,
    pi,
    residuo.
    Guarda en CSV y también imprime por consola.
    """
    headers = [
        "N",
        "montecarlo",
        "pi",
        "residuo"
    ]

    rows = []
    for N in N_values:
        estimate = montecarlo_pi(N)
        pi = np.pi
        remainder = estimate - np.pi
        rows.append([N, estimate, pi, remainder])

    # Guardar CSV.
    with open(file_name, "w", newline="") as file:
        my_writer = csv.writer(file, delimiter=";")
        my_writer.writerow(headers)
        for row in rows:
            N = row[0]
            formated_values = [f"{v:.8f}".replace(".", ",") for v in row[1:]]
            my_writer.writerow([N] + formated_values)

    # Imprimir por consola.
    print(f"{'N':>6} | {'montecarlo':>20} | {'pi':>20} | {'residuo':>20}")
    print("-" * 85)
    for row in rows:
        N, estimate, pi, remainder = row
        print(f"{N:>6} | {estimate:>20.8f} | {pi:>20.8f} | {remainder:>20.8f}")


N_values = sorted(set(
    list(range(10, 101, 10)) +
    list(range(200, 1001, 100)) +
    list(range(2000, 10001, 1000))
))

generate_table(N_values, "parte_4/tabla_montecarlo.csv")