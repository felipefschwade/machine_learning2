
from collections import Counter
import pandas as pd #Utiliza a biblioteca pandas de Data Analysis
df = pd.read_csv('buscas.csv') #Df = data_frame
Y_df = df['comprou'] #O pandas torna o cabecaalho como referencia ao dado, nao necessitando referencia numerica
X_df = df[['home', 'busca', 'logado']] #Para utilizar mais de uma coluna devemos passar um array como argumento
Xdummies_df= pd.get_dummies(X_df)
X = Xdummies_df.values
Y = Y_df.values
tamanho_de_treino = int(0.8 * len(X)) #define que o treino do algoritmo sera de 80 dos dados
tamanho_de_teste = int(0.1 * len(X)) #define que o teste do algoritmo sera de 10 dos dados
tamanho_de_validacao = int(0.1 * len(X)) #define que a validacao do algoritmo sera de 10 dos dados
#cria as variaveis de teste e de treino
treino_X = X[:tamanho_de_treino]
treino_Y = Y[:tamanho_de_treino]
fim_de_treino = tamanho_de_treino + tamanho_de_validacao
teste_X = X[tamanho_de_treino:fim_de_treino]
teste_Y = Y[tamanho_de_treino:fim_de_treino]
fim_de_teste = fim_de_treino + tamanho_de_treino
validacao_Y = Y[fim_de_treino:]
validacao_X = X [fim_de_treino:]
#cria a funcao de treinar e realizar os testes para determinado modelo
def fit_and_predict(nome, modelo, treino_X, treino_Y, teste_X, teste_Y):
	modelo.fit(treino_X, treino_Y) #realiza o treino #realiza o teste
	taxa_de_acerto = 100.0 * modelo.score(teste_X, teste_Y) #calcula a taxa de acerto
	msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)
	print msg
	#Testando a eficacia de um algoritmo que chuta tudo 0 ou 1
	acerto_base = max(Counter(teste_Y).itervalues())
	taxa_de_acerto_base = 100.0 * acerto_base / len(teste_Y)
	print "Taxa de acerto base: %f" % taxa_de_acerto_base
	return taxa_de_acerto

#importa a biblioteca do modelo bayesiano
from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_X, treino_Y, teste_X, teste_Y)
#importa a biblioteca do Adaboost
from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_X, treino_Y, teste_X, teste_Y)

if resultadoMultinomial > resultadoAdaBoost:
	vencedor = modeloMultinomial
else:
	vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_X)
taxa_de_acerto = 100.0 * vencedor.score(validacao_X, validacao_Y)
msg = "Taxa de acerto do algoritmo vencedor no mundo real: {0}".format(taxa_de_acerto)
print msg
	