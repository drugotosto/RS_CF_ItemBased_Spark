__author__ = 'maury'

import json
from collections import defaultdict
from pyspark import SparkContext

from tools.evaluator import Evaluator
from conf.confRS import topN, dirPathInput, datasetJSON, nFolds, percTestRates


class Recommender:
    def __init__(self,name):
        self.name=name
        self.topN=topN
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
        rdd=self.retrieveRatingsByUser(sc,datasetJSON)
        print("\nLettura File 'dataSet.json' e creazione RDD completata")
        fold=0
        # Ciclo sul numero di folds stabiliti andando a creare ogni volta i trainSetFold e testSetFold corrispondenti
        while fold<nFolds:

            """ Costruizione dell'RDD che costituirà il TestSetFold finale """
            trainUsers=sc.parallelize([item for sublist in usersFolds[:fold]+usersFolds[fold+1:] for item in sublist]).map(lambda x: (x,1))
            rddTestData=rdd.subtractByKey(trainUsers)
            # Costruisco un Pair RDD filtrato dei soli users appartenenti al dato fold del tipo (user,([(user,item,score),...,(user,item,score)],[(user,item,score),...,(user,item,score)]))
            test_trainParz=rddTestData.map(lambda item: Recommender.addUser(item[0],item[1])).map(lambda item: Recommender.divideTrain_Test(item[0],item[1],percTestRates))
            # Recupero la prima parte del campo Value di ogni elemento che rappresenta il TestSet del fold
            testSetData=test_trainParz.values().keys().flatMap(lambda x: x)

            """ Costruizione dell'RDD che costituirà il TrainSetFold finale """
            # Recupero la seconda parte del campo Value di ogni elemento che rappresenta una prima parte del TrainTest del fold
            trainSetData2=test_trainParz.values().values().filter(lambda x: x).flatMap(lambda x: x)
            testUsers=sc.parallelize(usersFolds[fold]).map(lambda x: (x,1))
            trainSetData1=rdd.subtractByKey(testUsers).map(lambda item: Recommender.addUser(item[0],item[1])).values().flatMap(lambda x: x)
            trainSetData=trainSetData1.union(trainSetData2).distinct()

            testSetData.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"test_"+str(fold))
            trainSetData.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"train_"+str(fold))
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

    def retrieveRatingsByUser(self,sc,fileName):
        """
        Creando la "matrice" dei Rates raggruppandoli secondo i vari utenti
        :param sc: SparkContext utilizzato
        :type sc: SparkContext
        :param fileName: Filename dal quale recuperare i dati
        :return: RDD (user,[(item,score),(item,score),...])
        """
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
                jsonLine=json.loads(line)
                test_ratings[jsonLine[0]].append((jsonLine[1],jsonLine[2]))

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


