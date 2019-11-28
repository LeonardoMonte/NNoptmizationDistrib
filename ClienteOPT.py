import numpy as np
import random
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import json
import socket
from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

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

def creatpopuMSRA(sizepopu, sizecromo):

    limit = np.sqrt(6 / float(sizecromo))
    chromossomos = [np.random.uniform(low=-limit, high=limit, size=sizecromo) for x in range(sizepopu)]
    bias = [np.random.uniform(low=-limit,high=limit,size=7) for x in range(sizepopu)] # change the value if change the sizechromo
    return chromossomos,bias

def GArun(population,bias, ite, train_index,test_index):

    tempototalinicioGA = time.time()
    scoresfinalGA = []
    chromofinaisGA = []

    scores = creatThreadPool(population,bias,train_index,test_index)

    for x in range(ite):

        population = newgentournement(scores, population)
        scores = creatThreadPool(population, bias, train_index, test_index)

        scoresfinalGA.append(scores[np.argmax(scores)])
        chromofinaisGA.append(population[np.argmax(scores)])

    tempototalfimGA = time.time()

    return scoresfinalGA, chromofinaisGA, tempototalfimGA - tempototalinicioGA

def creatThreadPool(pesos,bias,train_index,test_index):

    p1,p2 = np.split(pesos,2)
    b1,b2 = np.split(bias,2)

    PORT1 = 5000
    PORT2 = 5001
    SERVER_IP1 = '192.168.25.209'
    SERVER_IP2 = '192.168.25.209'

    t1 = ThreadWithReturnValue(target=sendClient,args=(p1,b1,train_index,test_index,PORT1,SERVER_IP1,))
    t2 = ThreadWithReturnValue(target=sendClient,args=(p2,b2,train_index,test_index,PORT2,SERVER_IP2,))
    t1.start()
    t2.start()
    retorno = t1.join()
    retorno2 = t2.join()

    return np.concatenate((retorno,retorno2))

def sendClient(pesos,bias,train_index,test_index,PORT,SERVER_IP):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP,PORT))

    pesos = [peso.tolist() for peso in pesos]
    bias = [b.tolist() for b in bias]

    tosend = json.dumps({"pesos":pesos,"bias":bias,"train_index":train_index.tolist(),"test_index":test_index.tolist()})

    s.send(tosend.encode())

    back = json.loads(s.recv(4096))
    retorno = back.get("back")

    return retorno

def sendaviso():

    PORT = 5000
    SERVER_IP = '192.168.25.209'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP,PORT))

    tosend = json.dumps({"stop":"stop"})
    s.send(tosend.encode())

def experimentosdistribuido(imagesnoclass, imagesclass, ite,iteKfold,popu,kvalue):

    cont = 0
    timeGA = []

    k = KFold(kvalue, True, 1)

    for x in range(iteKfold):

        for train_index, test_index in k.split(imagesnoclass):

            pesos,bias = np.array(creatpopuMSRA(popu, 306))

            scoresfinalGA, _, timetotalGA = GArun(pesos,bias, ite, train_index,test_index)
            print(scoresfinalGA)
            timeGA.append(timetotalGA)

            print("end")

        imagesnoclass, imagesclass = shuffle(imagesnoclass, imagesclass,
                                             random_state=cont)  # Depois que termina as 10 iterações de um Kfold dou shuffle
        cont += 1

    sendaviso()

    return timeGA

X,Y = loadcsv()

time1 = time.time()
timemedioGA = experimentosdistribuido(X,Y,10,2,20,10)
time2 = time.time()

print("Tempo total GA")
print(timemedioGA)
print("Tempo medio GA")
print(np.mean(timemedioGA))
print("Tempo total experimento distribuido")
print(time2-time1)