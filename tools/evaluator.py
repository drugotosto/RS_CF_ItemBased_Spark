__author__ = 'maury'

import json
from collections import defaultdict
from sklearn.metrics import mean_absolute_error,mean_squared_error

from conf.confRS import dirPath
from recommender import Recommender

class Evaluator:
    def __init__(self):
        # Dizionario che rappresenta i dati che compongono il TestSet (verrà settato più avanti)
        self.test_ratings=None

    def setTestRatings(self,rs,fold):
        """
        Costruisco il dizionario relativo al TestSet associato al dato fold preso in considerazione
        :param fold:
        :return:
        """
        # Costruisco un dizionario {user : [(item,rate),(item,rate),...] dai dati del TestSet
        fileName = dirPath+"testSetFold_"+str(fold)+".json"
        test_ratings=defaultdict(list)
        nTestRates=0
        with open(fileName) as f:
            for line in f.readlines():
                nTestRates+=1
                test_ratings[json.loads(line)[0]].append((json.loads(line)[1],json.loads(line)[2]))

        rs.appendNTestRates(nTestRates)
        self.test_ratings=test_ratings


    def computeEvaluation(self,rs):
        """
        Calcolo delle diverse misure di valutazione per il dato Recommender passato in input per un certo fold
        :param rs: Recommender da valutare
        :type rs: Recommender
        :return:
        """
        precisions=[]
        recalls=[]
        listMAEfold=[]
        listRMSEfold=[]
        # Numero di predizioni personalizzate che si è stati in grado di fare su tutto il fold
        nPredPers=0
        # Ciclo sul dizionario del test per recuperare le coppie (ratePred,rateTest)
        for userTest,ratingsTest in self.test_ratings.items():
            # Controllo se per il suddetto utente è possibile effettuare una predizione personalizzata
            if userTest in rs.getDictRec() and len(rs.getDictRec()[userTest])>0:
                # Coppie di (TrueRates,PredRates) preso in esame il tale utente
                pairsRatesPers=[]
                # Numero di items tra quelli ritenuti rilevanti dall'utente che sono stati anche fatti tornare
                numTorRil=0
                # Numero totale di items ritenuti rilevanti dall'utente
                nTotRil=0
                predRates,items=zip(*rs.getDictRec()[userTest])
                # Ciclo su tutti gli items per cui devo predire il rate
                for item,rate in ratingsTest:
                    # Controllo che l'item sia tra quelli per cui si è fatta una predizione
                    if item in items:
                        # Aggiungo la coppia (ScorePredetto,ScoreReale) utilizzata per MAE,RMSE
                        pairsRatesPers.append((predRates[items.index(item)],rate))
                        nPredPers+=1

                    # Controllo se l'item risulta essere rilevante
                    if rate>3:
                        nTotRil+=1
                        #  Controllo nel caso sia presente nei TopN suggeriti
                        if item in items[:rs.getTopN()]:
                            numTorRil+=1

                if pairsRatesPers:
                    # Calcolo MAE,RMSE (personalizzato) per un certo utente per tutti i suoi testRates
                    trueRates=[elem[0] for elem in pairsRatesPers]
                    predRates=[elem[1] for elem in pairsRatesPers]
                    mae=mean_absolute_error(trueRates,predRates)
                    listMAEfold.append(mae)
                    rmse=mean_squared_error(trueRates,predRates)
                    listRMSEfold.append(rmse)

                # Controllo se tra i rates dell'utente usati come testSet ci sono anche rates di items ritenuti Rilevanti
                if nTotRil>0:
                    # Calcolo della RECALL per il tale utente sotto esame
                    recalls.append(numTorRil/nTotRil)
                    # Calcolo della PRECISION per il tale utente sotto esame
                    precisions.append(numTorRil/rs.getTopN())

        # Registro le valutazioni appena calcolare per il fold preso in considerazione
        rs.appendMisuresFold(nPredPers,listMAEfold,listRMSEfold,recalls,precisions)

