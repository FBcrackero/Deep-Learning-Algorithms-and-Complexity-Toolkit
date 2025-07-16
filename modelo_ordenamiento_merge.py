import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# --- Entorno RL para Merge Sort (acciones: merge de sublistas adyacentes) ---
class MergeEnv:
    def __init__(self, lista, k=2):
        self.init_list = lista.copy()
        self.k = k  # tamaño mínimo de sublista a fusionar
        self.reset()

    def reset(self):
        self.lista = self.init_list.copy()
        self.steps = 0
        self.segments = [[i] for i in range(len(self.lista))]  # índices de segmentos actuales
        return self.get_state()

    def get_state(self):
        # Estado: la lista actual + segmentos actuales (aplanados)
        flat = []
        for seg in self.segments:
            flat += [self.lista[i] for i in seg]
        # Padding para tamaño fijo
        max_len = len(self.lista)
        flat += [0] * (max_len - len(flat))
        return np.array(flat, dtype=np.float32)

    def is_sorted(self):
        return all(self.lista[i] <= self.lista[i+1] for i in range(len(self.lista)-1))

    def available_actions(self):
        # Acciones: merge de segmentos adyacentes
        return [(i, i+1) for i in range(len(self.segments)-1)]

    def step(self, action):
        i, j = action
        # Fusionar segmentos i y j
        seg_i = self.segments[i]
        seg_j = self.segments[j]
        merged_indices = seg_i + seg_j
        merged_values = [self.lista[idx] for idx in merged_indices]
        merged_values.sort()
        # Actualizar la lista
        for offset, idx in enumerate(merged_indices):
            self.lista[idx] = merged_values[offset]
        # Actualizar segmentos
        self.segments = self.segments[:i] + [merged_indices] + self.segments[j+1:]
        self.steps += 1
        reward = -1
        done = False
        if self.is_sorted() and len(self.segments) == 1:
            reward = 100
            done = True
        return self.get_state(), reward, done

# --- Agente RL simple (DQN-like) ---
class QNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(n, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n-1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Entrenamiento RL con listas aleatorias ---
def train_rl(epochs=10000, max_steps=20, n=8):
    model_path = f"merge_rl_model_{n}.pth"
    model = QNet(n)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.9

    # Cargar modelo si existe
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Modelo cargado, continuará aprendiendo.")
    else:
        print("Entrenando desde cero.")

    for epoch in range(epochs):
        lista = random.sample(range(1, n+1), n)
        env = MergeEnv(lista)
        actions = env.available_actions()
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            actions = env.available_actions()
            if not actions:
                break
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            q_values = model(state_tensor).detach().numpy().flatten()
            # Epsilon-greedy
            if random.random() < 0.2:
                action_i = random.randint(0, len(actions)-1)
            else:
                action_i = np.argmax(q_values[:len(actions)])
            action = actions[action_i]
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-learning update
            next_state_tensor = torch.tensor(np.array([next_state]), dtype=torch.float32)
            target = reward
            if not done:
                next_actions = env.available_actions()
                if next_actions:
                    target += gamma * model(next_state_tensor)[:len(next_actions)].max().item()
            target_q = model(state_tensor).clone().detach()
            target_q[0, action_i] = target

            # Optimizar
            optimizer.zero_grad()
            output = model(state_tensor)
            loss = nn.MSELoss()(output, target_q)
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break
        if (epoch+1) % 1000 == 0:
            print(f"Epoca {epoch+1}, Lista: {lista}, Total reward: {total_reward}, Steps: {t+1}")

    # Guardar modelo entrenado
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

    return model, n

# --- Prueba ---
if __name__ == "__main__":
    while True:
        model, n = train_rl(epochs=10000, n=8)
        test_lista = random.sample(range(1, n+1), n)
        env = MergeEnv(test_lista)
        state = env.reset()
        print(f"Lista inicial: {state.tolist()}")
        for step in range(30):
            actions = env.available_actions()
            if not actions:
                print("No hay más merges posibles.")
                break
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            q_values = model(state_tensor).detach().numpy().flatten()
            action_i = np.argmax(q_values[:len(actions)])
            action = actions[action_i]
            state, reward, done = env.step(action)
            print(f"Paso {step+1}: merge segmentos {action}, lista: {state.tolist()}, reward: {reward}")
            if done:
                print("¡Lista ordenada!")
                break
        print("Ciclo de entrenamiento terminado. Iniciando otro ciclo...\n")