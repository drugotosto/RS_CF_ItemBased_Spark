__author__ = 'maury'

import time
import numpy as np
from numpy.random import choice

from conf.confRS import nFolds,categoria
from conf.confDirFiles import  datasetJSON, dirPathOutput, dirPathInput, dirTrain, dirTest
from conf.confItemBased import typeSimilarity
from recommenders.itemBased import ItemBased
from recommenders.social_itemBased import SocialItemBased
from tools.tools import saveJsonData, printRecVal
from sparkEnvLocal import SparkEnvLocal
from dataScienceAnalyzer import DataScienceAnalyzer

if __name__ == '__main__':
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Instanzio Recommender
    rs=ItemBased(name="ItemBased")
    # Instanzio Analizzatore dei Dati
    analyzer=DataScienceAnalyzer(categoria=categoria)
    """ Recupero dati da files (json) per la creazione del Dataframe finale e salvataggio su file 'dataSet.json' """
    dfMergeRB=analyzer.createDataSet()
    saveJsonData(dfMergeRB[["user_id","business_id","stars"]].values.tolist(),dirPathInput,datasetJSON)

    # **************** Effettuo diversi test costruendo per ognuno di essi Folds con users differenti ****************
    startTime=time.time()
    """ Creazione dei files (trainSetFold_k/testSetFold_k) per ogni prova di valutazione"""
    # SUDDIVISIONE DEGLI UTENTI IN K_FOLD (k=5)
    usersFolds=np.array_split(choice(dfMergeRB["user_id"].unique(),dfMergeRB["user_id"].unique().shape[0],replace=False),5)
    rs.createFolds(spEnv.getSc(),list(map(list,usersFolds)))
    print("\nCreazione files  all'interno delle cartelle 'Folds/train_k' -  'Folds/test_k' terminata!")

    fold=0
    # Ciclo su tutti i folds files (train/test)
    while fold<nFolds:

        """ Costruzione del modello a seconda dell'approccio utilizzato """
        rs.builtModel(spEnv.getSc(),dirTrain+str(fold))
        print("\nTempo impiegato per costruire il modello: {} sec.".format((time.time()-startTime)/60))
        """
        Ho calcolato tutte le possibile raccomandazioni personalizzare per i vari utenti con tanto di predizione per ciascun item
        Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
        """
        rs.retrieveTestData(dirTest+str(fold))
        rs.runEvaluation()
        print("\nEseguita Valutazione per Fold {}".format(fold))
        fold+=1
    """
    Salvataggio su file (json) dei risultati finali di valutazione (medie dei valori sui folds)
    """
    printRecVal(evaluator=rs.getEvaluator(),directory=dirPathOutput,fileName=dirPathOutput+rs.getName()+typeSimilarity)
    print("\nComputazione terminata! Durata della prova: {} min.".format((startTime-time.time())/60))

