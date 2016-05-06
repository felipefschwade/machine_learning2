#!-*- coding: utf-8 -*-
#importa a biblioteca numerica do python
import numpy as np
#importa o contador
from collections import Counter
import pandas as pd
#importa o natural language toolkit
import nltk
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
#cria um Stemmer utilizando o Removedor de Sufixos da Lingua Portuguesa para extrair a raiz das palavras
#nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()
#nltk.download('punkt') para utilizar o tokenize
#importa o metodo cross_val_score da biblioteca do cross_validation
from sklearn.cross_validation import cross_val_score 
classificacoes = pd.read_csv("emails.csv", encoding = 'utf-8')
textos_puros = classificacoes['email']
#quebra os textos em um conjunto de palavras separadas
textos_minusculos = textos_puros.str.lower()
textos_quebrados = [nltk.tokenize.word_tokenize(textos_minusculos) for textos_minusculos in textos_minusculos]
#cria um conjunto que não aceida dados iguais
dicionario = set()
#cria uma lista com os textos quebrados que não estão nas stopwords e adiciona ao conjunto
for lista in textos_quebrados:
	validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
	dicionario.update(validas)
#define a quantidade todal de palavras
total_de_palavras = len(dicionario)
#cria tuplas contendo a palavra e o seu indice
tuplas = zip(dicionario, xrange(total_de_palavras))
#cria um tradutor que irá mapear as palavras e os indices
tradutor = {palavra:indice for palavra,indice in tuplas}
#cria a função que transforma um texto em um vetor numerico
def vetoriza_texto(texto, tradutor):
	vetor = [0] * len(tradutor)
	for palavra in texto:
		if len(palavra) > 0: 
			raiz = stemmer.stem(palavra) 
			if raiz in tradutor:
				posicao = tradutor[stemmer.stem(palavra)]
				vetor[posicao] += 1
	return vetor

vetores_de_texto = [vetoriza_texto(texto, tradutor) for texto in textos_quebrados]
X = np.array(vetores_de_texto)
Y = np.array(classificacoes['classificacao'].tolist())
tamanho_de_treino = int(0.8 * len(X)) #define que o treino do algoritmo sera de 80 dos dados

treino_X = X[:tamanho_de_treino]
treino_Y = Y[:tamanho_de_treino]
validacao_X = X[tamanho_de_treino:]
validacao_Y = Y[tamanho_de_treino:]


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
