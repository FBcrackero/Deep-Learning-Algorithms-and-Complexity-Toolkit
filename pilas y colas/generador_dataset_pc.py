import numpy as np
import csv
import os
import random

def simular_estructura(ops, capacidad=5):
    stack = []
    queue = []
    pops = []
    for accion, val in ops:
        if accion == 0:  # push(x)
            if len(stack) < capacidad:
                stack.append(val)
        elif accion == 1:  # pop()
            if stack:
                pops.append(stack.pop())
            else:
                pops.append(-1)
        elif accion == 2:  # enqueue(x)
            if len(queue) < capacidad:
                queue.append(val)
        elif accion == 3:  # dequeue()
            if queue:
                pops.append(queue.pop(0))
            else:
                pops.append(-1)
    final_stack = stack + [0]*(capacidad - len(stack))
    final_queue = queue + [0]*(capacidad - len(queue))
    return pops, final_stack, final_queue

def generar_dataset_csv(ruta_csv, num_ejemplos=1200, max_len=6, max_val=20, capacidad=5):
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    with open(ruta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = []
        for i in range(max_len):
            header += [f"op{i}", f"val{i}"]
        header += [f"pop{i}" for i in range(max_len)]
        header += [f"stack{i}" for i in range(capacidad)]
        header += [f"queue{i}" for i in range(capacidad)]
        writer.writerow(header)
        # Datos
        for _ in range(num_ejemplos):
            seq_len = random.randint(2, max_len)
            ops = []
            for _ in range(seq_len):
                accion = random.choice([0, 1, 2, 3])
                val = random.randint(1, max_val) if accion in [0,2] else 0
                ops.append([accion, val])
            ops += [[1,0]]*(max_len - len(ops))
            pops, final_stack, final_queue = simular_estructura(ops, capacidad)
            pops += [0]*(max_len - len(pops))
            row = []
            for op, val in ops:
                row += [op, val]
            row += pops
            row += final_stack
            row += final_queue
            writer.writerow(row)

# Guardar dataset
csv_path = os.path.join(os.path.dirname(__file__), "dataset_pc.csv")
generar_dataset_csv(csv_path)
print(f"Dataset guardado en {csv_path}")