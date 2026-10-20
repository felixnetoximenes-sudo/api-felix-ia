# API FÉLIX - Inteligência Artificial para Predição de Dados

Esta é uma API comercial de alta performance desenvolvida para carregar e executar um modelo de aprendizado de máquina (Deep Learning) treinado no PyTorch. A API recebe dados numéricos de entrada através de requisições HTTP e retorna previsões em tempo real.

## 🚀 Tecnologias Utilizadas
* **Python** (Linguagem base)
* **PyTorch** (Criação e execução da rede neural)
* **FastAPI** (Construção da API HTTP de alta performance)
* **Pandas & Scikit-Learn** (Tratamento e divisão dos dados tabulares)
* **Uvicorn** (Servidor Web ASGI para rodar a aplicação)

## 📦 Estrutura do Projeto
* `main.py`: Código principal da API FastAPI.
* `modelo_ia.pth`: Arquivo binário contendo os pesos treinados do modelo.
* `requirements.txt`: Lista de dependências necessárias para rodar o projeto na nuvem.
* `README.md`: Documentação oficial do projeto.

## 🛠️ Como Funciona o Endpoint de Predição (`/prever`)
A API disponibiliza uma rota do tipo `POST` que espera receber uma lista contendo exatamente **11 valores numéricos** (características extraídas da planilha de dados).

### Exemplo de Requisição (JSON):
```json
{
  "valores_entrada": [1.0, 2.5, 0.0, 4.1, 5.0, 1.2, 0.8, 9.9, 3.3, 0.1, 7.4]
}
```

### Exemplo de Resposta (JSON):
```json
{
  "status": "sucesso",
  "previsao_ia": 42.87
}
```

## 🌍 Deploy
O projeto está configurado para deploy automatizado na plataforma **Render** integrado diretamente com este repositório do GitHub.
