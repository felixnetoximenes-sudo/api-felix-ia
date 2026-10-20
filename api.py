import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 1. Copiamos a MESMA estrutura exata da rede que você treinou
class Programa(nn.Module):
    def __init__(self):
        super().__init__()
        self.entrada = nn.Linear(11, 64) 
        self.ativacao = nn.ReLU()
        self.saida = nn.Linear(64, 1)

    def forward(self, x):
        x = self.entrada(x)
        x = self.ativacao(x)
        x = self.saida(x)
        return x

# 2. Inicializamos o FastAPI e carregamos o modelo
app = FastAPI(title="API FÉLIX")

modelo = Programa()
# Carrega o arquivo gerado pelo seu script de treino (.pth) na memória da API
modelo.load_state_dict(torch.load("modelo_ia.pth", map_location=torch.device('cpu')))
modelo.eval() # Configura o modelo para modo de teste/produção

# 3. Definimos o formato que a API exige (uma lista com os 11 números da planilha)
class RequisicaoDados(BaseModel):
    valores_entrada: List[float]

# 4. Rota GET: Apenas para checar se a API está online no navegador
@app.get("/")
def inicio():
    return {
        "status": "online",
        "mensagem": "API FÉLIX ativa! Use a rota /docs para mandar dados para a IA."
    }

# 5. Rota POST: Onde o usuário ENVIA OS DADOS PARA A IA calcular
@app.post("/prever")
def calcular_previsao(dados: RequisicaoDados):
    # Validação de segurança: garante que o usuário enviou exatamente 11 números
    if len(dados.valores_entrada) != 11:
        return {
            "status": "erro",
            "mensagem": f"A IA precisa de exatamente 11 valores numéricos. Você enviou {len(dados.valores_entrada)}."
        }
    
    # PASSO A PASSO DA IA:
    # A. Converte os números recebidos em um Tensor do PyTorch (Float32)
    tensor_entrada = torch.tensor([dados.valores_entrada], dtype=torch.float32)
    
    # B. Manda os dados para dentro da IA sem calcular gradientes (muito mais rápido)
    with torch.no_grad():
        previsao = modelo(tensor_entrada)
        
    # C. Pega o resultado numérico que a rede cuspiu e devolve para o usuário
    return {
        "status": "sucesso",
        "previsao_ia": float(previsao.item())
    }
