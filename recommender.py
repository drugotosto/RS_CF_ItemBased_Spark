__author__ = 'maury'

import json
from statistics import mean

from conf.confRS import topN,dirPath

class Recommender:

    def __init__(self):
        self.topN=topN
        self.dirPath=dirPath
         # Insieme di tutti i suggerimenti calcolari per i diversi utenti (verrà settato più avanti)
        self.dictRec=None
        # Risultati derivanti dalla valutazione medie delle diverse metriche utilizzate sui folds
        self.dataEval={"nTestRates":[],"nPredPers":[],"mae":[],"rmse":[],"precision":[],"recall":[],"f1":[]}


    def retriveData(self,sc,fold):
        """
        Recupero i dati (dai files) su cui poi andare a costruire il modello
        :return:
        """
        # Recupero i dati del TrainSet creando la conseguente "matrice" dei Rate raggruppando i rates dei vari utenti
        fileName = self.dirPath+"trainSetFold_"+str(fold)+".json"
        # Costruisco un pairRDD del tipo (user,[(item,rate),(item,rate),...]) e lo rendo persistente
        parseFile=self.parseFile
        user_item_pairs = sc.textFile(fileName).map(lambda line: parseFile(line)).groupByKey().cache()
        return user_item_pairs

    def parseFile(self,line):
        """
        Parsifico ogni linea del file e costruisco (user,(item,rate))
        """
        jsonObj = json.loads(line)
        return jsonObj[0],(jsonObj[1],float(jsonObj[2]))

    def builtModel(self,sc,rdd):
        """
        Costruzione del modello a seconda dell'approccio utilizzato (metodo astratto...)
        :return:
        """
        pass

    def appendNTestRates(self,nTestRates):
        self.dataEval["nTestRates"].append(nTestRates)

    def appendMisuresFold(self,nPredPers,listMAEfold,listRMSEfold,recalls,precisions):
        self.dataEval["nPredPers"].append(nPredPers)
        # Calcolo del valore medio di MAE,RMSE sui vari utenti appartenenti al fold
        # print("MAE (personalizzato) medio fold: {}".format(mean(listMAEfold)))
        self.dataEval["mae"].append(mean(listMAEfold))
        # print("RMSE (personalizzato) medio fold: {}".format(mean(listRMSEfold)))
        self.dataEval["rmse"].append(mean(listRMSEfold))
        # Calcolo del valore medio di precision e recall sui vari utenti appartenenti al fold
        # print("MEAN RECALL: {}".format(mean(recalls)))
        self.dataEval["recall"].append(mean(recalls))
        # print("MEAN PRECISION: {}".format(mean(precisions)))
        self.dataEval["precision"].append(mean(precisions))
        f1=(2*mean(recalls)*mean(precisions))/(mean(recalls)+mean(precisions))
        # print("F1 FOLD: {}".format(f1))
        self.dataEval["f1"].append(f1)

    def getTopN(self):
        return self.topN

    def setDictRec(self, dictRec):
        self.dictRec=dictRec

    def getDictRec(self):
        return self.dictRec

    def getDataEval(self):
        return self.dataEval

    def saveDataEval(self):
        """
        Salvataggio su file dei risultati derivanti dalla valutazione del recommender
        :return:
        """