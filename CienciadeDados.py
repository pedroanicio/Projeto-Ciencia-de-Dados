#install matplotlib
#install seaborn
#install scikit-learn

# importar a base de dados para o python
import pandas as pd
tabela = pd.read_csv("advertising.csv")
#print(tabela)
#  print(tabela.info())

# visualizar a base e fazer os ajustes necessários
# análise exploratória -> entender como a base está se comportando
# correlação entre tv -> vendas, rádio -> vendas e jornal -> vendas
# matplotlib e o seaborn para criar gráficos

import seaborn as sns
import matplotlib.pyplot as plt

# cria grafico
sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
#plt.show()

# treino e teste da ia

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

#Regressão Linear
#RandomForest (Árvore de Decisão)

# criar a IA e fazer as previsões
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# testar qual ia foi melhor

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# R² -> diz o % que o nosso modelo consegue explicar o que acontece

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))


#tabela_auxiliar = pd.DataFrame()
#tabela_auxiliar['y_teste'] = y_teste
#tabela_auxiliar['Previsoes arvore de decisao'] = previsao_arvoredecisao
#tabela_auxiliar['Previsoes regressao linear'] = previsao_regressaolinear

#plt.figure(figsize=(15,6))
#sns.lineplot(data=tabela_auxiliar)
#plt.show()

novos = pd.read_csv("novos.csv")

# modelo vencedor : arvore de decisão
previsao = modelo_arvoredecisao.predict(novos)
print(previsao)