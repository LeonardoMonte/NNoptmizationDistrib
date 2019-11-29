import numpy as np
import random
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import pandas as pd

def loadcsv():

    df_final = pd.read_csv('database2.csv')
    cols = list(df_final.columns)
    cols.remove('Class')
    cols.remove('Unnamed: 0')
    df_images_noclass = df_final[cols]
    df_images_class = df_final['Class']

    return np.array(df_images_noclass),np.array(df_images_class)

def crossover(pai1, pai2):
    beta = random.random()

    filho = pai1 + beta * (pai2 - pai1)

    filho = np.clip(filho, -1, 1)

    return filho

def mutation(filho):
    beta = random.random()

    filho = beta * filho
    filho = np.clip(filho, -1, 1)

    return filho

def newgentournement(score, population):
    newgen = []

    scorescopy = score.copy()
    index1 = np.argmax(scorescopy)
    scorescopy[index1] = -999999
    index2 = np.argmax(scorescopy)

    newgen.append(population[index1])
    newgen.append(population[index2])

    for x in range(int((len(population) - 2) / 2)):

        index1 = random.randint(0, len(population) - 1)
        index2 = random.randint(0, len(population) - 1)

        index3 = random.randint(0, len(population) - 1)
        index4 = random.randint(0, len(population) - 1)

        if score[index1] > score[index2]:
            pai1 = population[index1]
        else:
            pai1 = population[index2]

        if score[index3] > score[index4]:
            pai2 = population[index3]
        else:
            pai2 = population[index4]

        txcrs = random.uniform(0, 1)

        if txcrs <= 0.7:
            son1 = crossover(np.array(pai1), np.array(pai2))
            son2 = crossover(np.array(pai2), np.array(pai1))
        else:
            son1 = pai1
            son2 = pai2

        txmt = random.uniform(0, 1)

        if txmt <= 0.03:
            mutation(np.array(son1))
            mutation(np.array(son2))

        newgen.append(son1)
        newgen.append(son2)

    return np.array(newgen)

# arrayfitnessGA = []
#path = 'C:/Users/leopk/Documents/BCC/Sistemas Distribuidos/projeto/model.sav'

model = MLPClassifier(hidden_layer_sizes=(6),activation='relu',solver='lbfgs',max_iter=20)

def creatpopuMSRA(sizepopu, sizecromo):

    limit = np.sqrt(6 / float(sizecromo))
    chromossomos = [np.random.uniform(low=-limit, high=limit, size=sizecromo) for x in range(sizepopu)]
    bias = [np.random.uniform(low=-limit,high=limit,size=7) for x in range(sizepopu)] # change the value if change the sizechromo
    return chromossomos,bias

def fitnessGA(chromossome,bias, Xtrain, ytrain, Xval, yval):

    w1 = np.array(np.split(np.array(chromossome[0:len(chromossome) - 6]), 50))
    w2 = np.array(np.reshape(chromossome[len(chromossome) - 7:len(chromossome) - 1], (6, 1)))

    b1 = np.array(bias[:6])
    b2 = np.array(bias[-1])

    model.coefs_ = [w1,w2]
    model.intercepts_ = [b1,b2]

    model.fit(Xtrain, ytrain)
    retorno = model.score(Xval, yval)

    # if len(arrayfitnessGA) == 0:
    #     pickle.dump(model, open(path, 'wb'))
    #
    # if len(arrayfitnessGA) > 1:
    #     if retorno > arrayfitnessGA[np.argmax(arrayfitnessGA)]:
    #         pickle.dump(model, open(path, 'wb'))
    #
    # arrayfitnessGA.append(retorno)

    return retorno

def GArun(population,bias, ite, Xtrain, ytrain, Xval, yval):

    tempototalinicioGA = time.time()
    timefitness = []
    scoresGA = []
    scoresfinalGA = []
    chromofinaisGA = []

    timefitnessinicio = time.time()
    for popu,bia in zip(population,bias):
        scoresGA.append(fitnessGA(popu,bia, Xtrain, ytrain, Xval, yval))
    timefitnessfim = time.time()

    timefitness.append(timefitnessfim-timefitnessinicio)
    for x in range(ite):

        population = newgentournement(scoresGA, population)
        scoresGA = []

        timefitnessinicio = time.time()
        for popu, bia in zip(population, bias):
            scoresGA.append(fitnessGA(popu, bia, Xtrain, ytrain, Xval, yval))
        timefitnessfim = time.time()

        timefitness.append(timefitnessfim - timefitnessinicio)

        scoresfinalGA.append(scoresGA[np.argmax(scoresGA)])
        chromofinaisGA.append(population[np.argmax(scoresGA)])

    tempototalfimGA = time.time()

    return scoresfinalGA, chromofinaisGA, tempototalfimGA - tempototalinicioGA,timefitness

def experimentos(imagesnoclass, imagesclass, ite, itekfold, popu, kvalue):

    # ite = Quantidade de gerações dos algoritmos evolutivos
    # iteKfold Quantidade de iterações que vai ser executado o cross validation de k = 10
    cont = 0  # Seed para o shuffle da base de dados
    # popu = Tamanho da população

    timeGA = []  # Array para salvar os tempos de execução dos Algoritmos
    timefitnessarray = []
    k = KFold(kvalue, True, 1)  # Instancia da classe Kfold com k = 10

    for x in range(itekfold):  # For da quantidade de vezes que ira ser executado o cross validation de k = 10

        for train_index, test_index in k.split(imagesnoclass):  # For do Kfold com K = 10
            X_train, X_test = np.array(imagesnoclass)[train_index], np.array(imagesnoclass)[test_index]  # Separando o conjunto de Treino separado a cima em treino e validação
            y_train, y_test = np.array(imagesclass)[train_index], np.array(imagesclass)[test_index]  # Separando o conjunto de Treino separado a cima em treino e validação

            #X_trainK, X_valK, y_trainK, y_valK = train_test_split(X_train, y_train, test_size=0.2)

            ############ CREATE POPU ######

            pesos,bias = creatpopuMSRA(popu, 306)  # Criando a população inicial

            ############ CREATE POPU #########

            ###### GA ########################

            scoresfinalGA, _, timetotalGA,timefitness = GArun(pesos, bias, ite, X_train, y_train, X_test, y_test)
            print(scoresfinalGA)
            timeGA.append(timetotalGA)  # de iterações declaradas no inicio
            timefitnessarray.append(timefitness)
            print("end")

        imagesnoclass, imagesclass = shuffle(imagesnoclass, imagesclass,
                                             random_state=cont)  # Depois que termina as 10 iterações de um Kfold dou shuffle
        cont += 1

    return timeGA,timefitness

X,Y = loadcsv()
time1 = time.time()
timemedioGA,timefitness = experimentos(X,Y,70,2,10,10)
time2 = time.time()
print("tempo total dos experimentos")
print(time2-time1)
print("Time medio GA")
print(np.mean(timemedioGA))
print("Time total GA")
print(np.sum(timemedioGA))
print("Tempo total de fitness")
print(np.sum(timefitness))
print("Tempo fitness médio")
print(np.mean(timefitness))