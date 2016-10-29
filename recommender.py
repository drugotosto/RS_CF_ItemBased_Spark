__author__ = 'maury'

import json
from collections import defaultdict

from tools.evaluator import Evaluator
from conf.confRS import topN,dirPathInput,dirPathOutput

class Recommender:
    def __init__(self,name):
        self.name=name
        self.topN=topN
        self.dirPathInput=dirPathInput
        self.dirPathOutput=dirPathOutput
         # Insieme di tutti i suggerimenti calcolari per i diversi utenti (verrà settato più avanti)
        self.dictRec=None
        # Inizializzazione Evaluatore
        self.evaluator=Evaluator()

    def retrieveTrainData(self,sc,fold):
        """
        Recupero i dati (dai files) su cui poi andare a costruire il modello
        :return:
        """
        # Recupero i dati del TrainSet creando la conseguente "matrice" dei Rate raggruppando i rates dei vari utenti
        fileName = self.dirPathInput+"trainSetFold_"+str(fold)+".json"
        # Costruisco un pairRDD del tipo (user,[(item,rate),(item,rate),...]) e lo rendo persistente
        parseFileUser=self.parseFileUser
        user_item_pairs = sc.textFile(fileName).map(lambda line: parseFileUser(line)).groupByKey().cache()
        return user_item_pairs

    def parseFileUser(self,line):
        """
        Parsifico ogni linea del file e costruisco (user,(item,rate))
        :param line: Linea del file in input
        :return:
        """
        jsonObj = json.loads(line)
        return jsonObj[0],(jsonObj[1],float(jsonObj[2]))

    def builtModel(self,sc,rdd,fold):
        """
        Metodo astratto per la costruzione del modello a seconda dell'approccio utilizzato
        :param sc: SparkContext utilizzato
        :param rdd: RDD iniziale creato dal file in input
        :param fold: Fold preso in considerazione
        :return:
        """
        pass

    def retrieveTestData(self,fold):
        # Costruisco un dizionario {user : [(item,rate),(item,rate),...] dai dati del TestSet
        fileName = dirPathInput+"testSetFold_"+str(fold)+".json"
        test_ratings=defaultdict(list)
        nTestRates=0
        with open(fileName) as f:
            for line in f.readlines():
                nTestRates+=1
                test_ratings[json.loads(line)[0]].append((json.loads(line)[1],json.loads(line)[2]))

        self.evaluator.setTestRatings(test_ratings)
        self.evaluator.appendNtestRates(nTestRates)

    def runEvaluation(self):
        self.evaluator.computeEvaluation(self.dictRec,self.topN)

    def getName(self):
        return str(self.name)

    def getTopN(self):
        return self.topN

    def setDictRec(self, dictRec):
        self.dictRec=dictRec

    def getDictRec(self):
        return self.dictRec

    def getEvaluator(self):
        return self.evaluator

    def getDirPathOutput(self):
        return self.dirPathOutput