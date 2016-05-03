#importa a biblioteca numerica do python
import numpy as np
#importa o contador
from collections import Counter
#importa o metodo cross_val_score da biblioteca do cross_validation
from sklearn.cross_validation import cross_val_score 
import pandas as pd #Utiliza a biblioteca pandas de Data Analysis
df = pd.read_csv('situacao_do_cliente.csv') #Df = data_frame
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']] #Para utilizar mais de uma coluna devemos passar um array como argumento
Y_df = df['situacao'] #O pandas torna o cabecaalho como referencia ao dado, nao necessitando referencia numerica
Xdummies_df= pd.get_dummies(X_df)
X = Xdummies_df.values
Y = Y_df.values
tamanho_de_treino = int(0.8 * len(X)) #define que o treino do algoritmo sera de 80 dos dados
#tamanho_de_validacao = len(X) - tamanho_de_treino;
#cria as variaveis de teste e de treino
treino_X = X[:tamanho_de_treino]
treino_Y = Y[:tamanho_de_treino]
validacao_Y = Y[tamanho_de_treino:]
validacao_X = X [tamanho_de_treino:]
#cria a funcao de treinar e realizar os testes para determinado modelo
def fit_and_predict(nome, modelo, treino_X, treino_Y):
	k = 10
	resultado = cross_val_score(modelo, treino_X, treino_Y, cv = k);
	taxa_de_acerto = np.mean(resultado)
	print "A taxa de acerto do {0} foi de {1}".format(nome, taxa_de_acerto)
	return taxa_de_acerto

resultados = {}
#importa a bilbioteca do multiclass OneVsRest
from sklearn.multiclass import OneVsRestClassifier
#importa o LinearSVC utilizado no OVR
from sklearn.svm import LinearSVC
modeloOVR =  OneVsRestClassifier(LinearSVC(random_state=0)) 
resultadoOVR = fit_and_predict("OneVsRest", modeloOVR, treino_X, treino_Y)
resultados[resultadoOVR] = modeloOVR

#importa a bilbioteca do multiclass OneVsOne
from sklearn.multiclass import OneVsOneClassifier
modeloOVO =  OneVsOneClassifier(LinearSVC(random_state=0)) 
resultadoOVO = fit_and_predict("OneVsOne", modeloOVO, treino_X, treino_Y)
resultados[resultadoOVO] = modeloOVO

#importa a biblioteca do modelo bayesiano
from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_X, treino_Y)
#importa a biblioteca do Adaboost
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_X, treino_Y)
resultados[resultadoAdaBoost] = modeloAdaBoost
maximo = max(resultados)
vencedor = resultados[maximo]
print "O vencedor foi o modelo: {0}".format(vencedor)

vencedor.fit(treino_X, treino_Y)
resultado = vencedor.score(validacao_X, validacao_Y) * 100.0
msg = "Taxa de acerto do algoritmo vencedor no mundo real: {0}".format(resultado)
print msg
print "Quantidade de testes realizados: {0}".format(len(validacao_Y))
