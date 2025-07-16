import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import os

# Cargar el dataset con los 8 features
ruta = os.path.join(os.path.dirname(__file__), "complejidad_dataset.csv")
df = pd.read_csv(ruta)
X = torch.tensor(df[['for', 'while', 'if', 'recursion', 'logarithmic_pattern', 'max_loop_depth', 'multiple_recursion', 'loop_and_recursion']].values, dtype=torch.float32)
y_O = torch.tensor(df['O'].values, dtype=torch.long)
y_omega = torch.tensor(df['Omega'].values, dtype=torch.long)
y_theta = torch.tensor(df['Theta'].values, dtype=torch.long)

# Crear DataLoader
dataset = TensorDataset(X, y_O, y_omega, y_theta)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


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
        x = self.fc1(x)
        x = self.relu(x)
        out_O = self.fcO(x)
        out_omega = self.fcOmega(x)
        out_theta = self.fcTheta(x)
        return out_O, out_omega, out_theta

modelo = MultiComplexityNet()
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Entrenamiento vectorial por lotes
for epoca in range(501):
    correct_O = 0
    correct_omega = 0
    correct_theta = 0
    total = 0
    for xb, yb_O, yb_omega, yb_theta in loader:
        out_O, out_omega, out_theta = modelo(xb)
        perd_O = criterio(out_O, yb_O)
        perd_omega = criterio(out_omega, yb_omega)
        perd_theta = criterio(out_theta, yb_theta)
        perd = perd_O + perd_omega + perd_theta
        optimizador.zero_grad()
        perd.backward()
        optimizador.step()

        pred_O = torch.argmax(out_O, dim=1)
        pred_omega = torch.argmax(out_omega, dim=1)
        pred_theta = torch.argmax(out_theta, dim=1)
        correct_O += (pred_O == yb_O).sum().item()
        correct_omega += (pred_omega == yb_omega).sum().item()
        correct_theta += (pred_theta == yb_theta).sum().item()
        total += xb.size(0)

    if (epoca+1) % 100 == 0:
        acc_O = correct_O / total * 100
        acc_omega = correct_omega / total * 100
        acc_theta = correct_theta / total * 100
        print(f"Época {epoca+1}, Loss total: {perd.item():.4f} | Acc O: {acc_O:.2f}% | Acc Ω: {acc_omega:.2f}% | Acc Θ: {acc_theta:.2f}%")
torch.save(modelo.state_dict(), 'modelo_complejidad.pth')
print("Modelo guardado como 'modelo_complejidad.pth'")