__author__ = 'maury'

import os
import time
import numpy as np
import pandas as pd
from numpy.random import choice

from conf.confRS import nFolds, typeRecommender, numTags
from conf.confDirFiles import  datasetJSON, dirPathOutput, dirPathInput, dirTrain, dirTest, dirFolds
from conf.confItemBased import typeSimilarity, weightSim, nNeigh
from recommenders.itemBased import ItemBased
from tools.tools import saveJsonData, printRecVal
from tools.sparkEnvLocal import SparkEnvLocal
from tools.dataSetAnalyzer import DataScienceAnalyzer
from recommenders.socialBased import SocialBased
from recommenders.tagBased import TagBased
from conf.confCommunitiesFriends import communityType, communitiesTypes
from recommenders.tagSocialBased import TagSocialBased

if __name__ == '__main__':
    startTime=time.time()
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Instanzio Analizzatore dei Dati
    analyzer=DataScienceAnalyzer()

    if not os.path.exists(datasetJSON):
        os.makedirs(dirPathInput)
        """ Recupero dati da files (json) per la creazione del Dataframe finale e salvataggio su file 'dataSet.json' """
        analyzer.createDataSet(numTags=numTags)
        # Creazione File CSV
        analyzer.getDataFrame().to_csv(dirPathInput+"dataset.csv")
        # Creazione File dei ratings (filtrati) del Dataset
        saveJsonData(analyzer.getDataFrame()[["user_id","business_id","stars"]].values.tolist(),dirPathInput,datasetJSON)
        # Salvo il DataSet finale
        analyzer.getDataFrame().to_pickle(dirPathInput+"dataset")
    else:
        print("\n******** Il DataSet era già presente! *********")
        analyzer.setProperties(pd.read_pickle(dirPathInput+"dataset"))
        analyzer.printValuesDataset()
        numFriends=1
        print("Numero di utenti con almeno {} amici è: {}".format(numFriends,analyzer.getNumUsersWithFriends(numFriends)))
        # print("Distribuzione degli Utenti con almeno {} amici è : {}".format(numFriends,analyzer.getDistrFriends(numFriends)))

    rs=None
    # Instanzio il tipo di Recommender scelto
    if typeRecommender=="ItemBased":
        rs=ItemBased(name="ItemBased")
    elif typeRecommender=="TagBased":
        rs=TagBased(name="TagBased")
    elif typeRecommender=="SocialBased":
        friendships=analyzer.retrieveFriends()
        rs=SocialBased(name="SocialBased",friendships=friendships,communityType=communityType)
        rs.createFriendsCommunities()
    elif typeRecommender=="TagSocialBased":
        friendships=analyzer.retrieveFriends()
        rs=TagSocialBased(name="TagSocialBased",friendships=friendships)
        rs.createFriendsCommunities()

    if not os.path.exists(dirFolds):
        """ Creazione dei files (trainSetFold_k/testSetFold_k) per ogni prova di valutazione"""
        # SUDDIVISIONE DEGLI UTENTI IN K_FOLD
        usersFolds=np.array_split(choice(analyzer.getDataFrame()["user_id"].unique(),analyzer.getDataFrame()["user_id"].unique().shape[0],replace=False),nFolds)
        rs.createFolds(spEnv,list(map(list,usersFolds)))
        print("\nCreazione files  all'interno delle cartelle 'Folds/train_k' -  'Folds/test_k' terminata!")
    else:
        print("\nLe cartelle 'Folds/train_k' erano gia presenti!")

    fold=0
    # Ciclo su tutti i folds files (train/test)
    while fold<nFolds:
        """ Costruzione del modello a seconda dell'approccio utilizzato """
        rs.builtModel(spEnv,dirTrain+str(fold))
        print("\nModello costruito!")
        """
        Ho calcolato tutte le TOP-N raccomandazioni personalizzate per i vari utenti con tanto di predizione per ciascun item
        Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
        """
        rs.retrieveTestData(dirTest+str(fold))
        rs.getEvaluator().computeEvaluation(rs.getDictRec(),analyzer)
        print("\nEseguita Valutazione per Fold {}".format(fold))
        fold+=1

    """
    Salvataggio su file (json) dei risultati finali di valutazione (medie dei valori sui folds)
    """
    if not "Social" in rs.getName():
        fileName=dirPathOutput+rs.getName()+"/"+typeSimilarity+"_(nNeigh="+str(nNeigh)+",weightSim="+str(weightSim)+")"
    else:
        fileName=dirPathOutput+rs.getName()+"/"+communityType+"_"+typeSimilarity+"_(nNeigh="+str(nNeigh)+",weightSim="+str(weightSim)+")"
    printRecVal(evaluator=rs.getEvaluator(),directory=dirPathOutput+rs.getName()+"/",fileName=fileName)
    print("\nComputazione terminata! Durata totale: {} min.".format((time.time()-startTime)/60))

