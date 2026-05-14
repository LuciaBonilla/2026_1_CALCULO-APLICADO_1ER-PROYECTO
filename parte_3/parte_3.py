"""
SECCIÓN 3. Comparación de Métodos de Integración Numérica

En esta sección exploraremos distintos métodos de integración numérica sobre
una misma partición equiespaciada. Consideraremos 3 métodos numéricos
distintos: rectángulos, trapecio y punto medio.

1. Implemente en python funciones que tomen el número de puntos de una
partición equiespaciada en [-1, 1] y retornen la suma según los siguientes
métodos:
  - Rectángulos (el método usual)
  - Trapecio: en vez de áreas de rectángulos se suman áreas de trapecios
    definidos por los extremos de cada subintervalo.
  - Punto medio: evalúa la función en el centro de cada subintervalo.

2. Genere tablas comparativas de los 3 métodos variando el tamaño de la
partición. En esta tabla debe incluir los resultados de la aproximación y el
residuo (la diferencia entre la aproximación y el valor de π).
Una tabla debe variar el tamaño de N de 10 a 100 variando de a 10, la segunda
de 100 a 1000 variando de a 100 y la tercera de 1000 a 10000 variando de a 1000.

3. Grafique los valores de las 3 aproximaciones para distintos tamaños de N.
La gráfica debe incluir las 3 curvas y el valor de π teórico.

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


def rectangle_sum(N):
    """
    Aproximación de la integral de f en [-1, 1] por el método de rectángulos,
    con partición equiespaciada de N puntos. Se evalúa la función en el
    extremo izquierdo de cada subintervalo.
    """
    # Genera N números reales igualmente espaciados entre [-1, 1].
    x = np.linspace(-1.0, 1.0, N)

    # Base de los rectángulos de aproximación.
    delta = x[1] - x[0]

    # Evalúa f en cada extremo izquierdo: x_0, x_1, ..., x_{N-2}.
    f_vals = f(x[:-1])

    # Suma del área de los rectángulos.
    return delta * np.sum(f_vals)


def trapezoid_sum(N):
    """
    Aproximación de la integral de f en [-1, 1] por el método del trapecio,
    con partición equiespaciada de N puntos. Cada subintervalo aporta un
    trapecio cuyas alturas son f(x_i) y f(x_{i+1}).
    """
    x = np.linspace(-1.0, 1.0, N)
    delta = x[1] - x[0]
    f_vals = f(x)

    # Promedio de las alturas en los extremos de cada subintervalo:
    # area_i = ((f(x_i) + f(x_{i+1})) / 2) * delta.
    averages = (f_vals[:-1] + f_vals[1:]) / 2.0

    return delta * np.sum(averages)


def midpoint_sum(N):
    """
    Aproximación de la integral de f en [-1, 1] por el método del punto medio,
    con partición equiespaciada de N puntos. Se evalúa f en el centro de cada
    subintervalo.
    """
    x = np.linspace(-1.0, 1.0, N)
    delta = x[1] - x[0]

    # Centros de los subintervalos: (x_i + x_{i+1}) / 2.
    midpoints = (x[:-1] + x[1:]) / 2.0

    # Evalúa f en cada punto medio y suma las áreas.
    return delta * np.sum(f(midpoints))


def generate_table(N_values, file_name):
    """
    Genera una tabla con columnas:
    N,
    rectangulos,
    trapecios,
    puntos_medios,
    pi,
    residuo_rectangulos,
    residuo_trapecios,
    residuo_puntos_medios.
    Guarda en CSV y también imprime por consola.
    """
    headers = [
        "N",
        "rectangulos",
        "trapecios",
        "puntos_medios",
        "pi",
        "residuo_rectangulos",
        "residuo_trapecios",
        "residuo_puntos_medios"
    ]

    rows = []
    for N in N_values:
        R = rectangle_sum(N)
        T = trapezoid_sum(N)
        M = midpoint_sum(N)
        pi = np.pi
        remainder_R = R - np.pi
        remainder_T = T - np.pi
        remainder_M = M - np.pi
        rows.append([N, R, T, M, pi, remainder_R, remainder_T, remainder_M])

    # Guardar CSV.
    with open(file_name, "w", newline="") as file:
        my_writer = csv.writer(file, delimiter=";")
        my_writer.writerow(headers)
        for row in rows:
            N = row[0]
            formated_values = [f"{v:.8f}".replace(".", ",") for v in row[1:]]
            my_writer.writerow([N] + formated_values)

    # Imprimir por consola.
    print(f"{'N':>6} | {'rectangulos':>20} | {'trapecios':>20} | {'puntos_medios':>20} | {'pi':>20} | {'residuos_rectangulos':>20} | {'residuos_trapecios':>20} | {'residuos_puntos_medios':>20}")
    print("-" * 175)
    for row in rows:
        N, R, T, M, pi, remainder_R, remainder_T, remainder_M = row
        print(f"{N:>6} | {R:>20.8f} | {T:>20.8f} | {M:>20.8f} | {pi:>20.8f} | {remainder_R:>20.8f} | {remainder_T:>20.8f} | {remainder_M:>20.8f}")


N_values = sorted(set(
    list(range(10, 101, 10)) +
    list(range(200, 1001, 100)) +
    list(range(2000, 10001, 1000))
))

generate_table(N_values, "parte_3/tabla_comparacion_metodos.csv")