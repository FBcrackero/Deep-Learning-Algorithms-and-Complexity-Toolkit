import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os

# --- Parámetros ---
MAX_SEQ_LEN = 6
CAPACIDAD = 5
ACCIONES = 4  # push, pop, enqueue, dequeue
INPUT_DIM = ACCIONES + 1  # One-hot acción + valor

# --- Codificación one-hot de la acción ---
def codificar_operacion(accion, valor):
    one_hot = [0] * ACCIONES
    one_hot[accion] = 1
    return one_hot + [valor / 100]  # Normaliza valor entre 0 y 1

# --- Simulación (para dataset) ---
def simular_estructura(ops, capacidad=5):
    stack, queue, pops = [], [], []
    for accion, val in ops:
        if accion == 0:  # push(x)
            if len(stack) < capacidad:
                stack.append(val)
        elif accion == 1:  # pop()
            pops.append(stack.pop() if stack else -1)
        elif accion == 2:  # enqueue(x)
            if len(queue) < capacidad:
                queue.append(val)
        elif accion == 3:  # dequeue()
            pops.append(queue.pop(0) if queue else -1)
    final_stack = stack + [0]*(capacidad - len(stack))
    final_queue = queue + [0]*(capacidad - len(queue))
    return pops + [0]*(MAX_SEQ_LEN - len(pops)), final_stack, final_queue

# --- Dataset sintético ---
def generar_dataset_csv(ruta, muestras=10000):
    with open(ruta, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"op_{i}_{j}" for i in range(MAX_SEQ_LEN) for j in ["act", "val"]] + \
                 [f"pop_{i}" for i in range(MAX_SEQ_LEN)] + \
                 [f"stack_{i}" for i in range(CAPACIDAD)] + \
                 [f"queue_{i}" for i in range(CAPACIDAD)]
        writer.writerow(header)
        for _ in range(muestras):
            ops = [(np.random.randint(0, 4), np.random.randint(1, 100)) for _ in range(MAX_SEQ_LEN)]
            pops, stack, queue = simular_estructura(ops)
            row = [x for op in ops for x in op] + pops + stack + queue
            writer.writerow(row)

# --- Cargar dataset ---
def cargar_dataset_csv(ruta_csv):
    X, y_pops, y_stack, y_queue = [], [], [], []
    with open(ruta_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row = list(map(float, row))
            ops = [(int(row[2*i]), row[2*i+1]) for i in range(MAX_SEQ_LEN)]
            entrada = [codificar_operacion(a, v) for a, v in ops]
            X.append(entrada)
            offset = 2 * MAX_SEQ_LEN
            y_pops.append(row[offset:offset+MAX_SEQ_LEN])
            y_stack.append(row[offset+MAX_SEQ_LEN:offset+MAX_SEQ_LEN+CAPACIDAD])
            y_queue.append(row[offset+MAX_SEQ_LEN+CAPACIDAD:])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_pops, dtype=torch.float32),
            torch.tensor(y_stack, dtype=torch.float32),
            torch.tensor(y_queue, dtype=torch.float32))

# --- Modelo ---
class PilasColasNN(nn.Module):
    def __init__(self, input_size=INPUT_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.pop_fc = nn.Linear(hidden, 1)
        self.stack_fc = nn.Linear(hidden, CAPACIDAD)
        self.queue_fc = nn.Linear(hidden, CAPACIDAD)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        pop_out = self.relu(self.pop_fc(out)).squeeze(-1)
        stack_out = self.relu(self.stack_fc(out[:, -1]))
        queue_out = self.relu(self.queue_fc(out[:, -1]))
        return pop_out, stack_out, queue_out

# --- Entrenamiento ---
def entrenar():
    ruta_csv = "dataset_pc_onehot.csv"
    memoria_dir = os.path.join(os.path.dirname(__file__), "Memoria_PC")
    os.makedirs(memoria_dir, exist_ok=True)
    modelo_path = os.path.join(memoria_dir, "modelo_pilas_colas_final.pth")

    if not os.path.exists(ruta_csv):
        generar_dataset_csv(ruta_csv, 20000)

    X, y_pops, y_stack, y_queue = cargar_dataset_csv(ruta_csv)
    dataset = torch.utils.data.TensorDataset(X, y_pops, y_stack, y_queue)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    modelo = PilasColasNN()
    optimizador = optim.Adam(modelo.parameters(), lr=0.0005, weight_decay=1e-5)
    criterio = nn.MSELoss()

    for epoch in range(1, 1001):
        total_loss = 0
        modelo.train()
        for xb, yb_p, yb_s, yb_q in loader:
            pop_pred, stack_pred, queue_pred = modelo(xb)
            loss = criterio(pop_pred, yb_p) + criterio(stack_pred, yb_s) + criterio(queue_pred, yb_q)
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Época {epoch}, Pérdida total: {total_loss:.4f}")

    torch.save(modelo.state_dict(), modelo_path)
    print(f"Modelo guardado en {modelo_path}")

if __name__ == "__main__":
    entrenar()


