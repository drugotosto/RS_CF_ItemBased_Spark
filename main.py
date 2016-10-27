__author__ = 'maury'

from conf.confRS import nFolds
from recommenders.itemBased import ItemBased
from recommenders.social_itemBased import SocialItemBased
from tools.evaluator import Evaluator
from tools.printer import Printer
from sparkEnvLocal import SparkEnvLocal


if __name__ == '__main__':
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Inizializzazione Recommender
    rs=ItemBased()
    # Inizializzazione Evaluatore
    evaluator=Evaluator()
    # Inizializzazione Printer
    printer=Printer()
    k=0
    # Ciclo su tutti i folds files (train/test)
    while k<nFolds:
        # Recupero i dati (dai files) su cui poi andare a costruire il modello
        user_item_pairs=rs.retriveData(spEnv.getSc(),k)
        # Costruzione del modello a seconda dell'approccio utilizzato
        rs.builtModel(spEnv.getSc(),user_item_pairs)

        """
        Ho calcolato tutte le possibile raccomandazioni personalizzare per i vari utenti con tanto di predizione per ciascun item
        Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
        """
        evaluator.setTestRatings(rs,k)
        evaluator.computeEvaluation(rs)
        k+=1

    """
    Stampa e salvataggio su file dei risultati finali di valutazioni su tutti i folds (MEDIE valori)
    """
    printer.printRecVal(rs)
    # rs.saveDataEval()

