from flask import Flask, request, render_template
import torch
import torch.nn as nn
import sys
import os

# --- Importar analizar_codigo para complejidad ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "complejidad"))
from analizar_codigo import analizar_codigo

# --- Importar modelo RL de QuickSort ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ordenamiento"))
from modelo_ordenamiento_quick import QNet, QuickEnv

# --- Importar modelo de pilas y colas ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pilas y colas"))
from modelo_pilas_colas import PilasColasNN, codificar_operacion, MAX_SEQ_LEN, CAPACIDAD

app = Flask(__name__)

# --- Modelo de complejidad ---
class MultiComplexityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fcO = nn.Linear(16, 8)
        self.fcOmega = nn.Linear(16, 8)
        self.fcTheta = nn.Linear(16, 8)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=8, nhead=1, batch_first=True), num_layers=1
        )

    def forward(self, x):
        dummy = torch.zeros(x.size(0), 10, 8).to(x.device)
        _ = self.transformer(dummy)
        x = self.fc1(x)
        x = self.relu(x)
        out_O = self.fcO(x)
        out_omega = self.fcOmega(x)
        out_theta = self.fcTheta(x)
        return out_O, out_omega, out_theta

# Cargar modelo de complejidad
modelo_path = os.path.join(os.path.dirname(__file__), "complejidad", "MemoriaComp", "modelo_complejidad.pth")
modelo = MultiComplexityNet()
modelo.load_state_dict(torch.load(modelo_path, map_location=torch.device("cpu")))
modelo.eval()

clases = {
    0: "O(1)",
    1: "O(n)",
    2: "O(n^2)",
    3: "O(n^3)",
    4: "O(n^k)",
    5: "O(log n)",
    6: "O(n log n)",
    7: "O(n!)"
}

# --- Función para simular explícitamente las operaciones de pila y cola ---
def simular_estructura(ops, capacidad=5):
    stack, queue, pops = [], [], []
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

# --- Función para predecir estructuras de datos ---
def predecir(secuencia):
    return simular_estructura(secuencia, CAPACIDAD)

# --- Rutas Flask ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        codigo = request.form["codigo"]
        try:
            vector = analizar_codigo(codigo)
        except Exception as e:
            error = f"Código inválido: {e}"
            return render_template("index.html", error=error, codigo=codigo)
        if len(vector) != 8:
            error = f"El vector extraído tiene {len(vector)} features, pero el modelo espera 8."
            return render_template("index.html", error=error, codigo=codigo)
        entrada = torch.tensor([vector], dtype=torch.float32)
        with torch.no_grad():
            out_O, out_omega, out_theta = modelo(entrada)
            pred_O = torch.argmax(out_O, dim=1).item()
            pred_omega = torch.argmax(out_omega, dim=1).item()
            pred_theta = torch.argmax(out_theta, dim=1).item()
        return render_template("resultado.html", codigo=codigo, vector=vector,
                               O=clases[pred_O], omega=clases[pred_omega], theta=clases[pred_theta])
    return render_template("index.html")

@app.route("/quick", methods=["GET", "POST"])
def quicksort_rl():
    pasos = []
    lista_inicial = ""
    error = None
    if request.method == "POST":
        lista_str = request.form["lista"]
        try:
            lista = [int(x.strip()) for x in lista_str.split(",") if x.strip()]
        except Exception:
            error = "Por favor, ingresa una lista válida de números separados por coma."
            return render_template("quick.html", error=error, lista=lista_str)
        lista_inicial = lista_str
        n = len(lista)
        if n < 4 or n > 6:
            error = "El modelo RL solo soporta listas de tamaño 4, 5 o 6."
            return render_template("quick.html", error=error, lista=lista_str)
        modelo_quick_path = os.path.join("ordenamiento", "Memoria_Ordenamiento", f"quick_rl_model_{n}_target.pth")
        modelo_quick = QNet(n)
        modelo_quick.load_state_dict(torch.load(modelo_quick_path, map_location=torch.device("cpu")))
        modelo_quick.eval()
        env = QuickEnv(lista)
        state = env.reset()
        for step in range(30):
            actions = env.available_actions()
            if not actions:
                pasos.append({"step": step+1, "accion": "No hay más particiones posibles.", "lista": state.tolist()})
                break
            state_tensor = torch.tensor([state], dtype=torch.float32)
            q_values = modelo_quick(state_tensor).detach().numpy().flatten()
            action_i = int(torch.argmax(torch.tensor(q_values[actions])).item())
            action = actions[action_i]
            state, reward, done = env.step(action)
            pasos.append({"step": step+1, "accion": f"Pivote en índice {action}", "lista": state.tolist(), "reward": reward})
            if done:
                pasos.append({"step": step+1, "accion": "¡Lista ordenada!", "lista": state.tolist()})
                break
    return render_template("quick.html", pasos=pasos, lista=lista_inicial, error=error)

# --- Ruta para pilas y colas ---
@app.route("/pilas_colas", methods=["GET", "POST"])
def pilas_colas():
    resultado = None
    error = None
    secuencia = ""
    if request.method == "POST":
        secuencia = request.form["secuencia"]
        try:
            ops = []
            for par in secuencia.split(";"):
                if par.strip():
                    accion, valor = [int(x) for x in par.strip().split(",")]
                    ops.append((accion, valor))
            if len(ops) > MAX_SEQ_LEN:
                error = f"Máximo {MAX_SEQ_LEN} operaciones."
            else:
                pops, stack, queue = predecir(ops)
                resultado = {
                    "pops": pops,
                    "stack": stack,
                    "queue": queue
                }
        except Exception as e:
            error = f"Formato inválido. Usa: acción,valor;acción,valor;... ({str(e)})"
    return render_template("pilas_colas.html", resultado=resultado, secuencia=secuencia, error=error)

if __name__ == "__main__":
    app.run(debug=True)