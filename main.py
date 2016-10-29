__author__ = 'maury'

import time

from conf.confRS import nFolds
from conf.confItemBased import typeSimilarity
from recommenders.itemBased import ItemBased
from recommenders.social_itemBased import SocialItemBased
from tools.printer import Printer
from sparkEnvLocal import SparkEnvLocal


if __name__ == '__main__':
    # Instanzio Recommender
    rs=ItemBased(name="ItemBased")
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Instanzio Printer
    pr=Printer()
    k=0
    # Ciclo su tutti i folds files (train/test)
    while k<nFolds:
        startTime=time.time()
        # Recupero i dati (dai files) su cui poi andare a costruire il modello
        user_item_pairs=rs.retrieveTrainData(spEnv.getSc(),k)
        # Costruzione del modello a seconda dell'approccio utilizzato
        rs.builtModel(spEnv.getSc(),user_item_pairs,k)
        print("\nTempo impiegato per costruire il modello: {} sec.".format((time.time()-startTime)/60))
        """
        Ho calcolato tutte le possibile raccomandazioni personalizzare per i vari utenti con tanto di predizione per ciascun item
        Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
        """
        rs.retrieveTestData(k)
        rs.runEvaluation()
        k+=1

    """
    Salvataggio su file (testuale e binari) dei risultati finali di valutazioni su tutti i folds (MEDIE valori)
    """
    pr.printRecVal(evaluator=rs.getEvaluator(),directory=rs.getDirPathOutput(),fileName=rs.getDirPathOutput()+rs.getName()+typeSimilarity)

