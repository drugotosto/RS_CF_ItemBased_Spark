__author__ = 'maury'

import json
from collections import defaultdict

from conf.confDirFiles import *
from recommenders.itemBased import ItemBased
from recommenders.recommender import Recommender
from tools.sparkEnvLocal import SparkEnvLocal

class CommunitiesBased(ItemBased):
    def __init__(self,name):
        ItemBased.__init__(self,name=name)

    def builtModel(self,sc,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemBased
        :return:
        """
        """
        Calcolo del rate medio per ogni item e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (item,[(user,score),(user,score),...]
        item_user_pair=sc.textFile(directory+"/*").map(lambda line: Recommender.parseFileItem(line)).groupByKey()
        item_meanRates=item_user_pair.map(lambda p: CommunitiesBased.computeItemMean(p[0],p[1])).collectAsMap()
        dictItem_meanRates=sc.broadcast(item_meanRates)

        """
        Calcolo delle somiglianze tra items e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=sc.textFile(directory+"/*").map(lambda line: Recommender.parseFileUser(line)).groupByKey().cache()
        item_simsOrd=self.computeSimilarity(user_item_pair,dictItem_meanRates.value).collectAsMap()
        itemsSimil=sc.broadcast(item_simsOrd)

        """
        Calcolo delle (Top_N) raccomandazioni personalizzate per i diversi utenti
        """
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_item_pair.map(lambda p: CommunitiesBased.recommendations(p[0],p[1],itemsSimil.value,dictItem_meanRates.value,None)).map(lambda p: CommunitiesBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        print(self.dictRec)

    @staticmethod
    def recommendations(user_id,items_with_rating,item_sims,item_meanRates,friends):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        :param user_id: Utente per il quale si calcolano gli items da suggerire
        :param items_with_rating: Lista di items votati dall'utente
        :param item_sims: Matrice di somiglianza tra items calcolata precedentemente
        :param item_meanRates: Dizionario che contiene i rate medi per ogni item
        :return: Pair RDD del tipo: (user,[(scorePred,item),...])
        """
        # Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        for (item,rating) in items_with_rating:
            # Recupero tutti i vicini del tale item
            nearest_neighbors = item_sims.get(item,None)
            if nearest_neighbors:
                # Ciclo su tutti i vicini del tale item
                for (neighbor,(sim,listUsers)) in nearest_neighbors:
                    if neighbor != item:
                        # Aggiorno il valore di rate e somiglianza per il vicino in questione
                        totals[neighbor] += sim * (rating-item_meanRates.get(neighbor,None))
                        sim_sums[neighbor] += abs(sim)
        # Creo la lista dei rates normalizzati associati agli items per ogni user
        scored_items = [(item_meanRates.get(item,None)+(total/sim_sums[item]),item) for item,total in totals.items() if sim_sums[item]!=0.0]
        # Ordino la lista secondo il valore dei rates
        scored_items.sort(reverse=True)
        # Recupero i soli items
        # ranked_items = [x[1] for x in scored_items]
        return user_id,scored_items

    def createProjection(self,sc,fileName):
        # Recupero RDD di partenza
        rdd=sc.parallelize([(tag,user,val) for user,listPairs in json.load(open(fileName,"r")).items() for tag,val in listPairs])
        rdd=rdd.map(lambda x: (x[0], list(x[1:]))).groupByKey().values().map(lambda x: list(x))
        print(rdd.take(1))


if __name__ == '__main__':
    fileName="/home/maury/Desktop/ClusteringMethods/InputData/user_cat_(Nightlife)_2moreWeight.json"
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    rs=CommunitiesBased(name="CommunitiesBased")
    # Dalla rappresentazione del grafo bipartito di partenza costruisco la proiezione sugli utenti
    rs.createProjection(spEnv.getSc(),fileName)