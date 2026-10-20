import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 1. Ajustando a Rede para a realidade da planilha (11 entradas e 1 saída)
class Programa(nn.Module):
    def __init__(self):
        super().__init__()
        # Como a tabela tem 12 colunas no total, usaremos 11 para entrada e 1 para a resposta
        self.entrada = nn.Linear(11, 64) 
        self.ativacao = nn.ReLU()
        self.saida = nn.Linear(64, 1)

    def forward(self, x):
        x = self.entrada(x)
        x = self.ativacao(x)
        x = self.saida(x)
        return x

# 2. Lendo a planilha
tabela = pd.read_csv("calibrated_soft_labels.csv")

# TRATAMENTO DOS DADOS (Resolve o erro do TypeError):
# Converte tudo o que for possível para número. O que for texto vira "NaN" (vazio)
tabela = tabela.apply(pd.to_numeric, errors='coerce')
# Preenche qualquer valor vazio ou erro com 0.0 para não travar o PyTorch
tabela = tabela.fillna(0.0)

# 3. Fatiando os dados corretamente (11 colunas de entrada, a 12ª é o alvo)
valores = tabela.iloc[:, 0:11].values  # Colunas de 0 a 10 (11 colunas)
valores1 = tabela.iloc[:, 11].values   # Coluna de índice 11 (A 12ª coluna)

# Dividindo os dados
X_train, X_test, Y_train, Y_test = train_test_split(valores, valores1, test_size=0.2, random_state=42)

# Agora a conversão vai funcionar perfeitamente sem o erro de "object_"
dados_treino = torch.tensor(X_train, dtype=torch.float32)
item_treino = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

dados_teste = torch.tensor(X_test, dtype=torch.float32)
item_teste = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

# 4. Configuração do modelo e treino
modelo = Programa()
funcao_erro = nn.MSELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

for epoca in range(200):
    otimizador.zero_grad()
    previsao = modelo(dados_treino)
    perda = funcao_erro(previsao, item_treino)
    perda.backward()
    otimizador.step()
    
    if (epoca + 1) % 20 == 0:
        print(f"Época {epoca+1}/200 | Perda no Treino: {perda.item():.4f}")

# Avaliação final
modelo.eval()
with torch.no_grad():
    previsao_teste = modelo(dados_teste)
    perda_teste = funcao_erro(previsao_teste, item_teste)
    print(f"\n-> Perda nos dados de TESTE (Validação): {perda_teste.item():.4f}")
torch.save(modelo.state_dict(), "modelo_ia.pth")
print ("salvu")