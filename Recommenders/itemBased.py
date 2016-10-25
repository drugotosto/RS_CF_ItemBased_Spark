
__author__ = 'maury'

from itertools import combinations
import numpy as np
from scipy.spatial.distance import cosine
import operator
from collections import defaultdict

from recommender import Recommender
from conf.confItemBased import weightSim,nNeigh


class ItemBased(Recommender):

    def __int__(self,spEnv=None):
        """
        :param spEnv: Oggetto che rappresenta lo SparkEnviroment
        :return:
        """
        Recommender.__int__(self,spEnv)
        self.weightSim=weightSim
        self.nNeigh=nNeigh
        # Verrà settato più avanti
        self.dictRec=None

    def builtModel(self):
        """
        Costruzione del modello a secondo l'approccio CF ItemBased
        :return:
        """
        """
        Calcolo delle somiglianze con annessi valori per tutti gli elementi
        """
        item_simsOrd=self._computeSimilarity()

        """
        Creazione della broadcast variable da condividere con tutti i workers
        """
        item_sim_dict = {}
        for (item,data) in item_simsOrd.collect():
            item_sim_dict[item] = data
        itemsSimil = self.sc.broadcast(item_sim_dict)

        """
        Ho preprocessato i valori di somiglianza tra gli items e savati in una broadcast variable.
        Fornisco le raccomandazioni personalizzate (Top_N) per i diversi utenti
        """
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = self.user_item_pairs.map(lambda p: self._recommendations(p[0],p[1],itemsSimil.value)).map(lambda p: self._convertFloat_Int(p[0],p[1]))
        dictRec=dict(user_item_recs.collect())
        self.setDictRec(dictRec)


    def _findItemPairs(user_id,items_with_rating):
        """
        Ciclo su tutte le possibili combinazioni di item votati dall'utente restituendone le coppie con relativi rates
        """
        for item1,item2 in combinations(items_with_rating,2):
            return (item1[0],item2[0]),(item1[1],item2[1])


    def _calcSim(self,item_pair,rating_pairs):
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


    def _keyOnFirstItem(self,item_pair,item_sim_data):
        """
        Per ogni coppia di items faccio diventare il primo la key
        """
        (item1_id,item2_id) = item_pair
        return item1_id,(item2_id,item_sim_data)


    def _weightSim(self,item_id,items_and_sims):
        # items_and_simsWeight=[(item[1][1]/paramWeightSim)*item[1][0] if item[1][1]<paramWeightSim else item[1][1] for item in items_and_sims]
        items_and_simsWeight=list(map(self._weightSimElem,items_and_sims))
        return item_id,items_and_simsWeight


    def _weightSimElem(self,elem):
        return (elem[0], ((elem[1][1]/self.weightSim)*elem[1][0],elem[1][1])) if elem[1][1]<self.weightSim else elem


    def _nearestNeighbors(self,item_id,items_and_sims,n):
        """
        Ordino la lista di items più simili in base al valore di somiglianza recuperandone i primi n
        """
        s=sorted(items_and_sims,key=operator.itemgetter(1),reverse=True)
        return item_id, s[:n]

    def _removeOneRate(self,item_id,items_and_sims):
        """
        Rimuovo tutti quei items per i quali la lista delle somiglianze è <=1
        :param item_id: Item preso in considerazione
        :param items_and_sims: Items e associati valori di somiglianze per l'item sotto osservazione
        :return: Ritorno un nuovo pairRDD filtrato
        """
        lista=[item for item in items_and_sims if item[1][1]>1]
        if len(lista)>0:
            return item_id,lista

    def _computeSimilarity(self):
         # Costruisco un pairRDD del tipo ((item1,item2),[(rate1,rate2),(rate1,rate2),...])
        pairwise_items = self.user_item_pairs.filter(lambda p: len(p[1]) > 1).map(lambda p: self._findItemPairs(p[1])).groupByKey()
        # Costruisco un pairRDD del tipo (item,[(item1,(val_sim1,#rates1)),(item2,(val_sim2,#rates2),...]) dove i valori di somiglianza sono pesati in base al numero di Users in comune
        item_simsWeight = pairwise_items.map(lambda p: self._calcSim(p[0],p[1])).map(lambda p: self._keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: weightSim(p[0],p[1]))
        # Costruisco un pairRDD del tipo (item,[(item1,(val_sim1,#rates1)),(item2,(val_sim2,#rates2),...]) ordinato in base ai valori di somiglianza e filtrato (rimuovo tutte le somiglianze date da 1 solo utente)
        return item_simsWeight.map(lambda p: self._nearestNeighbors(p[0],p[1],self.nNeigh)).map(lambda p: self._removeOneRate(p[0],p[1])).filter(lambda p: p!=None)


    def _recommendations(self,user_id,items_with_rating,item_sims):
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



    def _convertFloat_Int(self,user_id,recommendations):
        """
        Converto da float a int il valore predetto per ogni elemento
        :param user_id: utente per il quale si vengono suggeriti gli items
        :param recommendations: lista di items raccomandati con associato valore (float)
        :return: user_id,recommendations
        """
        lista=[(int(pred),item) for pred,item in recommendations]
        return user_id,lista

    def setDictRec(self, dictRec):
        self.dictRec=dictRec
