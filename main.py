
__author__ = 'maury'

import json
from statistics import mean
from collections import defaultdict

from sklearn.metrics import mean_absolute_error,mean_squared_error

from conf.confRS import nFolds
from sparkEnvLocal import SparkEnvLocal
from Recommenders.itemBased import ItemBased
from Recommenders.social_itemBased import SocialItemBased

if __name__ == '__main__':
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Inizializzazione Recommender
    rs=ItemBased(spEnv)
    k=0
    listaMAE=[]
    listaRMSE=[]
    listaPrecision=[]
    listaRecall=[]
    listaF1=[]
    # Ciclo su tutti i folds files (train/test)
    while k<nFolds:
        # Recupero i dati (dai files) su cui poi andare a costruire il modello
        rs.retriveData(k)
        # Costruzione del modello a seconda dell'approccio utilizzato
        rs.builtModel()


        """
        Ho calcolato tutte le possibile predizioni personalizzare per i vari utenti con tanto di predizione per ciascun item
        Inzio la fase di valutazione utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
        """
        # Costruisco un dizionario {user : [(item,rate),(item,rate),...] dai dati del TestSet
        fileName = directoryPathSets+"testSetFold_"+str(k)+".json"
        test_ratings=defaultdict(list)
        nTestRates=0
        with open(fileName) as f:
            for line in f.readlines():
                nTestRates+=1
                test_ratings[json.loads(line)[0]].append((json.loads(line)[1],json.loads(line)[2]))

        """
        Calcolo delle varie Metriche (MAE,RMSE,PRECISION,RECALL) x il Fold
        """
        precisions=[]
        recalls=[]
        listMAEfold=[]
        listRMSEfold=[]
        # Numero di predizioni personalizzate che si è stati in grado di fare su tutto il fold
        nPredPers=0
        # Ciclo sul dizionario del test per recuperare le coppie (ratePred,rateTest)
        for userTest,ratingsTest in test_ratings.items():
            # Controllo se per il suddetto utente è possibile effettuare una predizione personalizzata
            if userTest in dictRec and len(dictRec[userTest])>0:
                # Coppie di (TrueRates,PredRates) preso in esame il tale utente
                pairsRatesPers=[]
                # Numero di items tra quelli ritenuti rilevanti dall'utente che sono stati anche fatti tornare
                numTorRil=0
                # Numero totale di items ritenuti rilevanti dall'utente
                nTotRil=0
                predRates,items=zip(*dictRec[userTest])
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
                        if item in items[:topN]:
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
                    precisions.append(numTorRil/topN)

        # print(pairsRatesPers)
        print("Predizioni totali da dover calcolare: {} - Predizioni personali calcolate: {}".format(nTestRates,nPredPers))
        # Calcolo del valore medio di MAE,RMSE sui vari utenti appartenenti al fold
        print("MAE (personalizzato) medio fold: {}".format(mean(listMAEfold)))
        listaMAE.append(mean(listMAEfold))
        print("RMSE (personalizzato) medio fold: {}".format(mean(listRMSEfold)))
        listaRMSE.append(mean(listRMSEfold))
        # Calcolo del valore medio di precision e recall sui vari utenti appartenenti al fold
        print("MEAN RECALL: {}".format(mean(recalls)))
        listaRecall.append(mean(recalls))
        print("MEAN PRECISION: {}".format(mean(precisions)))
        listaPrecision.append(mean(precisions))
        f1=(2*mean(recalls)*mean(precisions))/(mean(recalls)+mean(precisions))
        print("F1 FOLD: {}".format(f1))
        listaF1.append(f1)
        k+=1

    """
    Risultati finali di valutazioni su tutti i fold (MEDIE valori)
    """
    print("\n\n************ Risultati Finali *************")
    print("MAE: {}".format(mean(listaMAE)))
    print("RMSE: {}".format(mean(listaRMSE)))
    print("Precision: {}".format(mean(listaPrecision)))
    print("Recall: {}".format(mean(listaRecall)))
    print("F1: {}".format(mean(listaF1)))


