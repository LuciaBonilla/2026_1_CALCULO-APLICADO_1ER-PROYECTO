"""
SECCIÓN 2. Influencia del Tipo de Partición

1. Implemente en python funciones que tomen la cantidad de puntos de la
partición de [-1, 1] y generen una de las 3 siguientes particiones:
  - Equiespaciada
  - Partición aleatoria uniforme
  - Partición de la forma x_i = cos((i*pi)/N) para N tamaño de la partición

2. Genere tablas comparativas de las 3 particiones variando el tamaño de la
partición (usamos sumas inferiores).
En esta tabla debe incluir los resultados de la aproximación y el residuo (la
diferencia entre la aproximación y el valor de π) para cada suma.
Una tabla debe variar el tamaño de N de 10 a 100 variando de a 10, la segunda
de 100 a 1000 variando de a 100 y la tercera de 1000 a 10000 variando de a 1000.

3. Grafique los valores de las 3 sumas para las distintas particiones. La gráfica
debe incluir las 3 curvas y el valor de π teórico.

4. Grafique la función f y los rectángulos de la aproximación para cada partición,
con N = 100.
"""

import numpy as np
import csv


def f(x):
    """f(x) = 2*sqrt(1 - x^2)"""
    return 2.0 * np.sqrt(1.0 - x**2)


# Generador aleatorio reproducible (mismo resultado en cada ejecución).
rng = np.random.default_rng(seed=42)


def equispaced_partition(N):
    """
    N puntos equiespaciados en [-1, 1].
    """
    return np.linspace(-1.0, 1.0, N)


def random_partition(N):
    """
    N puntos en [-1, 1] obtenidos por muestreo uniforme aleatorio.
    Los extremos -1 y 1 se fijan; los N-2 puntos interiores se sortean
    uniformemente y se ordenan.
    """
    interior = rng.uniform(-1.0, 1.0, size=N - 2)
    return np.concatenate(([-1.0], np.sort(interior), [1.0]))


def cosine_partition(N):
    """
    N puntos de la forma x_i = cos((i*pi)/N) en [-1, 1]:
        x_i = cos((i*pi)/N) para i = 0, ..., N-1
    Se invierten para que queden en orden creciente (de -1 a 1).
    """
    i = np.arange(N)
    return np.flip(np.cos((i * np.pi) / N))


def lower_sum(x):
    """
    Suma inferior de Riemann de f sobre la partición x (no necesariamente
    equiespaciada).
    """
    widths = np.diff(x)                       # x_{i+1} - x_i
    f_vals = f(x)
    m = np.minimum(f_vals[:-1], f_vals[1:])
    return np.sum(widths * m)


def generate_table(N_values, file_name):
    """
    Tabla con columnas:
    N,
    sum_inf_equiespaciada,
    sum_inf_aleatoria,
    sum_inf_coseno,
    pi,
    residuo_equiespaciada,
    residuo_aleatoria,
    residuo_coseno.
    """
    headers = [
        "N",
        "sum_inf_equiespaciada",
        "sum_inf_aleatoria",
        "sum_inf_coseno",
        "pi",
        "residuo_equiespaciada",
        "residuo_aleatoria",
        "residuo_coseno"
    ]

    rows = []
    for N in N_values:
        L_eq = lower_sum(equispaced_partition(N))
        L_rand = lower_sum(random_partition(N))
        L_cos = lower_sum(cosine_partition(N))
        pi = np.pi
        remainder_eq = L_eq - np.pi
        remainder_rand = L_rand - np.pi
        remainder_cos = L_cos - np.pi
        rows.append([N, L_eq, L_rand, L_cos, pi, remainder_eq, remainder_rand, remainder_cos])

    # Guardar CSV.
    with open(file_name, "w", newline="") as file:
        my_writer = csv.writer(file, delimiter=";")
        my_writer.writerow(headers)
        for row in rows:
            N = row[0]
            formated_values = [f"{v:.8f}".replace(".", ",") for v in row[1:]]
            my_writer.writerow([N] + formated_values)

    # Imprimir por consola.
    print(f"{'N':>6} | {'sum_inf_equiespaciada':>20} | {'sum_inf_aleatoria':>20} | {'sum_inf_coseno':>20} | {'pi':>20} | {'residuo_equiespaciada':>20} | {'residuo_aleatoria':>20} | {'residuo_coseno':>20}")
    print("-" * 175)
    for row in rows:
        N, L_eq, L_rand, L_cos, pi, remainder_eq, remainder_rand, remainder_cos = row
        print(f"{N:>6} | {L_eq:>20.8f} | {L_rand:>20.8f} | {L_cos:>20.8f} | {pi:>20.8f} | {remainder_eq:>20.8f} | {remainder_rand:>20.8f} | {remainder_cos:>20.8f}")


N_values = sorted(set(
    list(range(10, 101, 10)) +
    list(range(200, 1001, 100)) +
    list(range(2000, 10001, 1000))
))


# ── Gráfico: función f y rectángulos de suma inferior para cada partición (N = 100) ──

import matplotlib.pyplot as plt

def graph_approximation_rectangles():
    N_plot = 100
    x_curve = np.linspace(-1.0, 1.0, 1000)

    partitions = [
        equispaced_partition(N_plot),
        random_partition(N_plot),
        cosine_partition(N_plot)
    ]
    titles = ["Equiespaciada", "Aleatoria uniforme", "Coseno"]
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, x, title, color in zip(axes, partitions, titles, colors):
        # Anchos y alturas de los rectángulos de la suma inferior.
        widths = np.diff(x)
        f_vals = f(x)
        heights = np.minimum(f_vals[:-1], f_vals[1:])

        ax.bar(
            x[:-1], heights, width=widths, align="edge",
            color=color, alpha=0.4, edgecolor=color, linewidth=0.3,
            label="Suma inferior"
        )
        ax.plot(x_curve, f(x_curve), color="black", linewidth=1.5, label="f(x)", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.1, 2.3)
        ax.legend(loc="upper center")

    axes[0].set_ylabel("f(x)")
    plt.suptitle(f"Suma inferior por tipo de partición — N = {N_plot} puntos", fontsize=13)
    plt.tight_layout()
    plt.savefig("parte_2/grafico_particiones_N100.png", dpi=150)
    plt.show()



generate_table(N_values, "parte_2/tabla_influencia_particion.csv")
graph_approximation_rectangles()