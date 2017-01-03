from recommenders.tagSocialBased import TagSocialBased

__author__ = 'maury'

import time
import numpy as np
from numpy.random import choice

from conf.confRS import nFolds, typeRecommender, numTags
from conf.confDirFiles import  datasetJSON, dirPathOutput, dirPathInput, dirTrain, dirTest
from conf.confItemBased import typeSimilarity, weightSim
from recommenders.itemBased import ItemBased
from tools.tools import saveJsonData, printRecVal
from tools.sparkEnvLocal import SparkEnvLocal
from tools.dataSetAnalyzer import DataScienceAnalyzer
from recommenders.socialBased import SocialBased
from recommenders.tagBased import TagBased

if __name__ == '__main__':
    startTime=time.time()
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Instanzio Analizzatore dei Dati
    analyzer=DataScienceAnalyzer()

    """ Recupero dati da files (json) per la creazione del Dataframe finale e salvataggio su file 'dataSet.json' """
    analyzer.createDataSet(numTags=numTags)
    # Creazione File CSV
    analyzer.dataFrame.to_csv(dirPathInput+"Dataset.csv")
    # Creazione File dei ratings (filtrati) del Dataset
    saveJsonData(analyzer.getDataFrame()[["user_id","business_id","stars"]].values.tolist(),dirPathInput,datasetJSON)

    rs=None
    # Instanzio il tipo di Recommender scelto
    if typeRecommender=="ItemBased":
        rs=ItemBased(name="ItemBased")
    elif typeRecommender=="TagBased":
        rs=TagBased(name="TagBased")
    elif typeRecommender=="SocialBased":
        friendships=analyzer.retrieveFriends()
        rs=SocialBased(name="SocialBased",friendships=friendships)
        rs.createFriendsCommunities()
    elif typeRecommender=="TagSocialBased":
        friendships=analyzer.retrieveFriends()
        rs=TagSocialBased(name="TagSocialBased",friendships=friendships)
        rs.createFriendsCommunities()

    """ Creazione dei files (trainSetFold_k/testSetFold_k) per ogni prova di valutazione"""
    # SUDDIVISIONE DEGLI UTENTI IN K_FOLD
    usersFolds=np.array_split(choice(analyzer.getDataFrame()["user_id"].unique(),analyzer.getDataFrame()["user_id"].unique().shape[0],replace=False),nFolds)
    rs.createFolds(spEnv,list(map(list,usersFolds)))
    print("\nCreazione files  all'interno delle cartelle 'Folds/train_k' -  'Folds/test_k' terminata!")

    fold=0
    # Ciclo su tutti i folds files (train/test)
    while fold<nFolds:
        """ Costruzione del modello a seconda dell'approccio utilizzato """
        rs.builtModel(spEnv,dirTrain+str(fold))
        print("\nModello costruito!")
        """
    #     Ho calcolato tutte le possibile raccomandazioni personalizzare per i vari utenti con tanto di predizione per ciascun item
    #     Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
    #     """
    #     rs.retrieveTestData(dirTest+str(fold))
    #     rs.getEvaluator().computeEvaluation(rs.getDictRec(),analyzer)
    #     # print("\nEseguita Valutazione per Fold {}".format(fold))
        fold+=1
    #
    # """
    # Salvataggio su file (json) dei risultati finali di valutazione (medie dei valori sui folds)
    # """
    # printRecVal(evaluator=rs.getEvaluator(),directory=dirPathOutput,fileName=dirPathOutput+rs.getName()+typeSimilarity+"_(numTags="+str(numTags)+",weightSim="+str(weightSim)+")")
    # print("\nComputazione terminata! Durata totale: {} min.".format((time.time()-startTime)/60))
    #
