__author__ = 'maury'

import os
import json
import shutil
from os import listdir
from collections import defaultdict
from pyspark import SparkContext

from tools.evaluator import Evaluator
from conf.confDirFiles import datasetJSON, dirFolds, dirTest, dirTrain
from conf.confRS import nFolds, percTestRates


class Recommender:
    def __init__(self,name):
        self.name=name
         # Insieme di tutti i suggerimenti calcolari per i diversi utenti (verrà settato più avanti)
        self.dictRec=None
        # Inizializzazione Evaluatore
        self.evaluator=Evaluator()


    def createFolds(self,sc,usersFolds):
        """
        Creazione dei diversi TestSetFold/TrainTestFold partendo dal DataSet presente su file
        :param sc: SparkContext utilizzato
        :type sc: SparkContext
        :param usersFolds: Lista di folds che separano i diversi utenti
        :return:
        """
        # Elimino la cartella che contiene i diversi files che rappresentano i diversi folds
        if os.path.exists(dirFolds):
            shutil.rmtree(dirFolds)
        # Creo la "matrice" dei Rates raggruppandoli secondo i vari utenti ottengo (user,[(item,score),(item,score),...])
        user_item_pair=sc.textFile(datasetJSON).map(lambda line: Recommender.parseFileUser(line)).groupByKey()
        fold=0
        # Ciclo sul numero di folds stabiliti andando a creare ogni volta i trainSetFold e testSetFold corrispondenti
        while fold<nFolds:

            """ Costruizione dell'RDD che costituirà il TestSetFold finale """
            trainUsers=sc.parallelize([item for sublist in usersFolds[:fold]+usersFolds[fold+1:] for item in sublist]).map(lambda x: (x,1))
            rddTestData=user_item_pair.subtractByKey(trainUsers)
            # Costruisco un Pair RDD filtrato dei soli users appartenenti al dato fold del tipo (user,([(user,item,score),...,(user,item,score)],[(user,item,score),...,(user,item,score)]))
            test_trainParz=rddTestData.map(lambda item: Recommender.addUser(item[0],item[1])).map(lambda item: Recommender.divideTrain_Test(item[0],item[1],percTestRates))
            # Recupero la prima parte del campo Value di ogni elemento che rappresenta il TestSet del fold
            testSetData=test_trainParz.values().keys().flatMap(lambda x: x)

            """ Costruizione dell'RDD che costituirà il TrainSetFold finale """
            # Recupero la seconda parte del campo Value di ogni elemento che rappresenta una prima parte del TrainTest del fold
            trainSetData2=test_trainParz.values().values().filter(lambda x: x).flatMap(lambda x: x)
            testUsers=sc.parallelize(usersFolds[fold]).map(lambda x: (x,1))
            trainSetData1=user_item_pair.subtractByKey(testUsers).map(lambda item: Recommender.addUser(item[0],item[1])).values().flatMap(lambda x: x)
            trainSetData=trainSetData1.union(trainSetData2).distinct()

            testSetData.map(lambda x: json.dumps(x)).saveAsTextFile(dirTest+str(fold))
            trainSetData.map(lambda x: json.dumps(x)).saveAsTextFile(dirTrain+str(fold))
            fold+=1

    @staticmethod
    def addUser(user,item_with_rating):
        newList=[(user,elem[0],elem[1]) for elem in item_with_rating]
        return user,newList

    @staticmethod
    def divideTrain_Test(user_id,user_item_rating,percTestRates):
        numElTest=int(len(user_item_rating)*percTestRates)
        if numElTest<1:
            numElTest=1
        dati=[elem for elem in user_item_rating]
        return user_id,(dati[:numElTest],dati[numElTest:])

    @staticmethod
    def parseFileUser(line):
        """
        Parsifico ogni linea del file stabilendo come chiave lo user
        :param line: Linea del file in input
        :return: (user,(item,rate))
        """
        jsonObj = json.loads(line)
        return jsonObj[0],(jsonObj[1],float(jsonObj[2]))

    @staticmethod
    def parseFileItem(line):
        """
        Parsifico ogni linea del file stabilendo come chiave l'item
        :param line: Linea del file in input
        :return (item,(user,rate))
        """
        jsonObj = json.loads(line)
        return jsonObj[1],(jsonObj[0],float(jsonObj[2]))

    def builtModel(self,sc,directory):
        """
        Metodo astratto per la costruzione del modello a seconda dell'approccio utilizzato
        :param sc: SparkContext utilizzato
        :param rdd: RDD iniziale creato dal file in input
        :return:
        """
        pass

    def retrieveTestData(self,directory):
        # Costruisco un dizionario {user : [(item,rate),(item,rate),...] dai dati del TestSet
        files=[file for file in listdir(directory) if file.startswith("part-")]
        test_ratings=defaultdict(list)
        nTestRates=0
        for fileName in files:
            with open(directory+"/"+fileName) as f:
                for line in f.readlines():
                    nTestRates+=1
                    jsonLine=json.loads(line)
                    test_ratings[jsonLine[0]].append((jsonLine[1],jsonLine[2]))

        self.evaluator.setTestRatings(test_ratings)
        self.evaluator.appendNtestRates(nTestRates)

    def getName(self):
        return str(self.name)

    def setDictRec(self, dictRec):
        self.dictRec=dictRec

    def getDictRec(self):
        return self.dictRec

    def getEvaluator(self):
        return self.evaluator


