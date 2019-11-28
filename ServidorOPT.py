import numpy as np
import time
from sklearn.neural_network import MLPClassifier
import pandas as pd
import socket
import json
from json import JSONDecodeError

model = MLPClassifier(hidden_layer_sizes=(6),activation='relu',solver='lbfgs',max_iter=20)

def loadcsv():

    df_final = pd.read_csv('database2.csv')
    cols = list(df_final.columns)
    cols.remove('Class')
    cols.remove('Unnamed: 0')
    df_images_noclass = df_final[cols]
    df_images_class = df_final['Class']

    return np.array(df_images_noclass),np.array(df_images_class)

def startconec(listensize,PORT,IP):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((IP, PORT))
    s.listen(listensize)

    return s

def traintest(train_index,test_index,imagesnoclass,imagesclass):

    X_train, X_test = np.array(imagesnoclass)[train_index], np.array(imagesnoclass)[test_index]
    y_train, y_test = np.array(imagesclass)[train_index], np.array(imagesclass)[test_index]

    return X_train,X_test,y_train,y_test

def fitnessGA(chromossome,bias, Xtrain, ytrain, Xval, yval):

    w1 = np.array(np.split(np.array(chromossome[0:len(chromossome) - 6]), 50))
    w2 = np.array(np.reshape(chromossome[len(chromossome) - 7:len(chromossome) - 1], (6, 1)))

    b1 = np.array(bias[:6])
    b2 = np.array(bias[-1])

    model.coefs_ = [w1,w2]
    model.intercepts_ = [b1,b2]

    model.fit(Xtrain, ytrain)
    retorno = model.score(Xval, yval)

    return retorno

save = []


def wait(s,noclasse,classe):

    while 1:
        try:

            conn, addr = s.accept()
            recev = json.loads(conn.recv(9000000))
            #if "stop" in recev: break


            pesos = recev.get("pesos")
            bias = recev.get("bias")
            train_index = recev.get("train_index")
            test_index = recev.get("test_index")

            X_train,X_test,y_train,y_test = traintest(train_index,test_index,noclasse,classe)

            score = []
            for p,b in zip(pesos,bias):
                score.append(fitnessGA(p,b,X_train,y_train,X_test,y_test))

            save.append(score)
            print(score)
            data = json.dumps({"back":score})
            conn.send(data.encode())


        except JSONDecodeError as ex:

            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(len(save[0]))
            data = json.dumps({"back":save[0]})
            conn.send(data.encode())
            #wait(s, noclasse, classe)

    conn.close()



noclasse,classe = loadcsv()
s = startconec(1000, 5001, '192.168.43.206')
wait(s,noclasse,classe)
