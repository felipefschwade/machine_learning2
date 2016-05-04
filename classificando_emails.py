#!-*- coding: utf-8 -*-
import pandas as pd
#importa o metodo cross_val_score da biblioteca do cross_validation
from sklearn.cross_validation import cross_val_score 
classificacoes = pd.read_csv("emails.csv")
textos_puros = classificacoes['email']
#quebra os textos em um conjunto de palavras separadas
textos_quebrados = textos_puros.str.lower().str.split(' ')
#cria um conjunto que não aceida dados iguais
dicionario = set()
#cria uma lista com os textos quebrados e adiciona ao conjunto
for lista in textos_quebrados:
	dicionario.update(lista)
#define a quantidade todal de palavras
total_de_palavras = len(dicionario)
#cria tuplas contendo a palavra e o seu indice
tuplas = zip(dicionario, xrange(total_de_palavras))
#cria um tradutor que irá mapear as palavras e os indices
tradutor = {palavra:indice for palavra,indice in tuplas}
#cria a função que transforma um texto em um vetor numerico
def vetoriza_texto(texto, tradutor):
	vetor = [0] * len(tradutor)
	texto = textos_quebrados[0]
	for palavra in texto:
		if palavra in tradutor:
			posicao = tradutor[palavra]
			vetor[posicao] += 1
	return vetor
vetores_de_texto = [] * len(tradutor)
for texto in textos_quebrados:
	vetores_de_texto = vetoriza_texto(texto, tradutor)
marcacoes = classificacoes['classificacao']
X = vetores_de_texto
Y = marcacoes
print X
print Y
tamanho_de_treino = int(0.8 * len(X)) #define que o treino do algoritmo sera de 80 dos dados
treino_X = X[:tamanho_de_treino]
treino_Y = Y[:tamanho_de_treino]
validacao_X = X[tamanho_de_treino:]
validacao_Y = Y[tamanho_de_treino:]
def fit_and_predict(nome, modelo, treino_X, treino_Y):
	k = 4
	resultado = cross_val_score(modelo, treino_X, treino_Y, cv = k);
	taxa_de_acerto = np.mean(resultado)
	print "A taxa de acerto do {0} foi de {1}".format(nome, taxa_de_acerto)
	return taxa_de_acerto
#importa a bilbioteca do multiclass OneVsOne
from sklearn.multiclass import OneVsOneClassifier
#importa o LinearSVC utilizado no OVR
from sklearn.svm import LinearSVC
modeloOVR =  OneVsOneClassifier(LinearSVC(random_state=0)) 
fit_and_predict("OneVsOneClassifier", modeloOVR, treino_X, treino_Y)