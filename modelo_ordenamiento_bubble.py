import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

# --- Entorno RL para Bubble Sort (solo swaps adyacentes) ---
class BubbleEnv:
    def __init__(self, lista):
        self.init_list = lista.copy()
        self.reset()

    def reset(self):
        self.lista = self.init_list.copy()
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return np.array(self.lista, dtype=np.float32)

    def is_sorted(self):
        return all(self.lista[i] <= self.lista[i+1] for i in range(len(self.lista)-1))

    def step(self, action):
        i = action
        # Swap adyacente
        self.lista[i], self.lista[i+1] = self.lista[i+1], self.lista[i]
        self.steps += 1
        reward = -1
        done = False
        if self.is_sorted():
            reward = 100
            done = True
        return self.get_state(), reward, done

    def available_actions(self):
        n = len(self.lista)
        return list(range(n-1))  # Solo swaps adyacentes

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

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action_i, reward, next_state, done):
        self.buffer.append((state, action_i, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_i, reward, next_state, done = map(np.array, zip(*batch))
        return state, action_i, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# --- Entrenamiento RL con Replay Buffer y listas pequeñas ---
def train_rl(epochs=10000, max_steps=20, n=5, batch_size=32):
    model_path = f"bubble_rl_model_{n}_replay.pth"
    model = QNet(n)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.9
    buffer = ReplayBuffer(capacity=5000)

    # Cargar modelo si existe
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Modelo cargado, continuará aprendiendo.")
    else:
        print("Entrenando desde cero.")

    for epoch in range(epochs):
        lista = random.sample(range(1, n+1), n)
        env = BubbleEnv(lista)
        actions = env.available_actions()
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            q_values = model(state_tensor).detach().numpy().flatten()
            # Epsilon-greedy
            if random.random() < 0.2:
                action_i = random.randint(0, len(actions)-1)
            else:
                action_i = np.argmax(q_values)
            action = actions[action_i]
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Guardar experiencia en el buffer
            buffer.push(state, action_i, reward, next_state, done)

            # Entrenamiento desde el buffer
            if len(buffer) >= batch_size:
                b_state, b_action_i, b_reward, b_next_state, b_done = buffer.sample(batch_size)
                b_state_tensor = torch.tensor(b_state, dtype=torch.float32)
                b_next_state_tensor = torch.tensor(b_next_state, dtype=torch.float32)
                b_action_i_tensor = torch.tensor(b_action_i, dtype=torch.long)
                b_reward_tensor = torch.tensor(b_reward, dtype=torch.float32)
                b_done_tensor = torch.tensor(b_done, dtype=torch.bool)

                q_values_batch = model(b_state_tensor)
                q_value = q_values_batch.gather(1, b_action_i_tensor.reshape(-1,1)).squeeze()

                with torch.no_grad():
                    next_q_values = model(b_next_state_tensor).max(1)[0]
                    target = b_reward_tensor + gamma * next_q_values * (~b_done_tensor)

                loss = nn.MSELoss()(q_value, target)

                optimizer.zero_grad()
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

    return model, actions, n

# --- Prueba ---
if __name__ == "__main__":
    while True:
        # n pequeño para facilitar el aprendizaje
        model, actions, n = train_rl(epochs=10000, n=random.randint(4,6))
        test_lista = random.sample(range(1, n+1), n)
        env = BubbleEnv(test_lista)
        state = env.reset()
        print(f"Lista inicial: {state.tolist()}")
        for step in range(30):
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            q_values = model(state_tensor).detach().numpy().flatten()
            action_i = np.argmax(q_values)
            action = actions[action_i]
            state, reward, done = env.step(action)
            print(f"Paso {step+1}: swap posiciones {action} y {action+1}, lista: {state.tolist()}, reward: {reward}")
            if done:
                print("¡Lista ordenada!")
                break
        print("Ciclo de entrenamiento terminado. Iniciando otro ciclo...\n")