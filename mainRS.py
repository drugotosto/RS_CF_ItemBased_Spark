__author__ = 'maury'
import json
import operator
from statistics import mean
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_absolute_error,mean_squared_error

from pyspark import SparkContext, SparkConf
from paramConfigApp import directoryPathSets
from conf.confRS import paramWeightSim, nNeighbours, topN, nFolds


def parseFile(line):
    """
    Parsifico ogni linea del file e costruisco (user,(item,rate))
    """
    jsonObj = json.loads(line)
    return jsonObj[0],(jsonObj[1],float(jsonObj[2]))
l
def findItemPairs(user_id,items_with_rating):
    """
    Ciclo su tutte le possibili combinazioni di item votati dall'utente restituendone le coppie con relativi rates
    """
    for item1,item2 in combinations(items_with_rating,2):
        return (item1[0],item2[0]),(item1[1],item2[1])

def calcSim(item_pair,rating_pairs):
    """
    Per ogni coppia di items ritorno il valore di somiglianza con annessi numero di voti in comune
    N.B: ATTENZIONE! Quando la coppia di item è stata votata da un solo user il valore di somiglianza è sempre di 1 (a prescindere dai rates)
    """
    item1Rates,item2Rates=zip(*rating_pairs)
    item1Rates=np.array(item1Rates)
    item2Rates=np.array(item2Rates)
    cos_sim=1-cosine(item1Rates,item2Rates)
    nComuneRates=item1Rates.shape[0]
    return item_pair, (cos_sim,nComuneRates)

def keyOnFirstItem(item_pair,item_sim_data):
    """
    Per ogni coppia di items faccio diventare il primo la key
    """
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def weightSim(item_id,items_and_sims):
    # items_and_simsWeight=[(item[1][1]/paramWeightSim)*item[1][0] if item[1][1]<paramWeightSim else item[1][1] for item in items_and_sims]
    items_and_simsWeight=list(map(weightSimElem,items_and_sims))
    return item_id,items_and_simsWeight

def weightSimElem(elem):
    return (elem[0], ((elem[1][1]/paramWeightSim)*elem[1][0],elem[1][1])) if elem[1][1]<paramWeightSim else elem

def nearestNeighbors(item_id,items_and_sims,n):
    """
    Ordino la lista di items più simili in base al valore di somiglianza recuperandone i primi n
    """
    s=sorted(items_and_sims,key=operator.itemgetter(1),reverse=True)
    return item_id, s[:n]

def removeOneRate(item_id,items_and_sims):
    """
    Rimuovo tutti quei items per i quali la lista delle somiglianze è <=1
    :param item_id: Item preso in considerazione
    :param items_and_sims: Items e associati valori di somiglianze per l'item sotto osservazione
    :return: Ritorno un nuovo pairRDD filtrato
    """
    lista=[item for item in items_and_sims if item[1][1]>1]
    if len(lista)>0:
        return item_id,lista

def recommendations(user_id,items_with_rating,item_sims):
    """
    Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
    N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
    :param user_id: Utente per il quale si calcolano gli items da suggerire
    :param items_with_rating: Lista di items votati dall'utente
    :param item_sims: Matrice di somiglianza tra items calcolata precedentemente
    :return: Lista di items suggeriti per ogni user preso in considerazione
    """
    # Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori
    totals = defaultdict(int)
    sim_sums = defaultdict(int)
    for (item,rating) in items_with_rating:
        # Recupero tutti i vicini del tale item
        nearest_neighbors = item_sims.get(item,None)
        if nearest_neighbors:
            # Ciclo su tutti i vicini del tale item
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item:
                    # Aggiorno il valore di rate e somiglianza per il vicino in questione
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim
    # Creo la lista normalizzata dei rates associati agli items
    scored_items = [(total/sim_sums[item],item) for item,total in totals.items()]
    # Ordino la lista secondo il valore dei rates
    scored_items.sort(reverse=True)
    # Recupero i soli items
    # ranked_items = [x[1] for x in scored_items]
    return user_id,scored_items

def convertFloat_Int(user_id,recommendations):
    """
    Converto da float a int il valore predetto per ogni elemento
    :param user_id: utente per il quale si vengono suggeriti gli items
    :param recommendations: lista di items raccomandati con associato valore (float)
    :return: user_id,recommendations
    """
    lista=[(int(pred),item) for pred,item in recommendations]
    return user_id,lista


if __name__ == '__main__':
    # Inizializzazione dello SparkContext
    conf = SparkConf().setMaster("local[8]").setAppName("RS_CF_ITEM-BASED")
    sc = SparkContext(conf=conf)
    k=0
    listaMAE=[]
    listaRMSE=[]
    listaPrecision=[]
    listaRecall=[]
    listaF1=[]
    # Ciclo su tutti i folds files (train/test)
    while k<nFolds:
        fileName = directoryPathSets+"trainSetFold_"+str(k)+".json"
        # Recupero i dati del TrainSet creando la conseguente "matrice" dei Rate raggruppando i rates dei vari utenti
        lines = sc.textFile(fileName)
        # Costruisco un pairRDD del tipo (user,[(item,rate),(item,rate),...])
        user_item_pairs = lines.map(parseFile).groupByKey().cache()
        # Costruisco un pairRDD del tipo ((item1,item2),[(rate1,rate2),(rate1,rate2),...])
        pairwise_items = user_item_pairs.filter(lambda p: len(p[1]) > 1).map(lambda p: findItemPairs(p[0],p[1])).groupByKey()
        # Costruisco un pairRDD del tipo (item,[(item1,(val_sim1,#rates1)),(item2,(val_sim2,#rates2),...]) dove i valori di somiglianza sono pesati in base al numero di Users in comune
        item_simsWeight = pairwise_items.map(lambda p: calcSim(p[0],p[1])).map(lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: weightSim(p[0],p[1]))
        # Costruisco un pairRDD del tipo (item,[(item1,(val_sim1,#rates1)),(item2,(val_sim2,#rates2),...]) ordinato in base ai valori di somiglianza e filtrato (rimuovo tutte le somiglianze date da 1 solo utente)
        item_simsOrd=item_simsWeight.map(lambda p: nearestNeighbors(p[0],p[1],nNeighbours)).map(lambda p: removeOneRate(p[0],p[1])).filter(lambda p: p!=None)

        # Creazione della broadcast variable da condividere con tutti i workers
        item_sim_dict = {}
        for (item,data) in item_simsOrd.collect():
            item_sim_dict[item] = data
        itemsSimil = sc.broadcast(item_sim_dict)

        """
        Ho preprocessato i valori di somiglianza tra gli items e savati in una broadcast variable.
        Inizio la fase di Valutazione del Recommender e raccomandazione personalizzata (Top_N) per i diversi utenti
        """

        # Cancello la directory che contiene i risultati se presente
        # outputPathDirectory="/home/maury/Desktop/SparkOutputsRecommedations/fold"+str(k)
        # if os.path.exists(outputPathDirectory):
        #     shutil.rmtree(outputPathDirectory)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_item_pairs.map(lambda p: recommendations(p[0],p[1],itemsSimil.value)).map(lambda p: convertFloat_Int(p[0],p[1]))
        dictRec=dict(user_item_recs.collect())
        # user_item_recs.saveAsTextFile(outputPathDirectory)

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

