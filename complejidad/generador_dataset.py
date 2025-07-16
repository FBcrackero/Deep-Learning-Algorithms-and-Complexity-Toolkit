from analizar_codigo import analizar_codigo
import csv

ejemplos = [
    # (codigo, O, Omega, Theta)
    # Constante
    ("""def acceso_constante(lista):
    return lista[0]
    """, 0, 0, 0),
    # Lineal
    ("""def suma_lineal(lista):
    total = 0
    for x in lista:
        total += x
    return total
    """, 1, 1, 1),
    # Cuadrática
    ("""def burbuja(lista):
    n = len(lista)
    for i in range(n):
        for j in range(n):
            if lista[i] > lista[j]:
                lista[i], lista[j] = lista[j], lista[i]
    return lista
    """, 2, 2, 2),
    # Cúbica
    ("""def triple_for(lista):
    n = len(lista)
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count += 1
    return count
    """, 3, 3, 3),
    # Polinomial (n^4)
    ("""def polinomial(lista):
    n = len(lista)
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    count += 1
    return count
    """, 4, 4, 4),
    # Logarítmica
    ("""def busqueda_binaria(lista, objetivo):
    inicio = 0
    fin = len(lista) - 1
    while inicio <= fin:
        medio = (inicio + fin) // 2
        if lista[medio] == objetivo:
            return True
        elif lista[medio] > objetivo:
            fin = medio - 1
        else:
            inicio = medio + 1
    return False
    """, 5, 0, 5),
    # Linealítica (n log n)
    ("""def merge_sort(lista):
    if len(lista) > 1:
        mid = len(lista) // 2
        L = lista[:mid]
        R = lista[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                lista[k] = L[i]
                i += 1
            else:
                lista[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            lista[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            lista[k] = R[j]
            j += 1
            k += 1
    return lista
    """, 6, 5, 6),
    # Factorial
    ("""def permutaciones(lista):
    if len(lista) == 0:
        return [[]]
    res = []
    for i in range(len(lista)):
        m = lista[i]
        remList = lista[:i] + lista[i+1:]
        for p in permutaciones(remList):
            res.append([m] + p)
    return res
    """, 7, 7, 7),
    # Exponencial (fibonacci recursivo)
    ("""def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    """, 7, 7, 7),
    # Otras variantes...
]

# Puedes aumentar el dataset duplicando y variando los ejemplos:
ejemplos_ampliados = ejemplos * 500  # 10 ejemplos * 50 = 500

dataset = []
for codigo, o, omega, theta in ejemplos_ampliados:
    feats = analizar_codigo(codigo)
    dataset.append(feats + [o, omega, theta])

with open("complejidad_dataset.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "for", "while", "if", "recursion",
        "logarithmic_pattern", "max_loop_depth", "multiple_recursion", "loop_and_recursion",
        "O", "Omega", "Theta"
    ])
    writer.writerows(dataset)

print(f"✅ ¡Dataset generado con {len(dataset)} ejemplos reales usando analizar_codigo!")