__author__ = 'maury'

from collections import defaultdict

from conf.confDirFiles import userTagJSON
from recommenders.itemBased import ItemBased
from recommenders.recommender import Recommender
from tools.sparkEnvLocal import SparkEnvLocal


class TagBased(ItemBased):
    def __init__(self,name):
        ItemBased.__init__(self,name=name)

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemBased
        :param spEnv: SparkContext di riferimento
        :type spEnv: SparkEnvLocal
        :param directory: Directory che contiene insieme di File che rappresentano il TestSet
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: Recommender.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: TagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo media dei valori associati ai Tags (per ogni user) e creazione della corrispondente broadcast variable (Utilizzo per misura Pearson)
        """
        user_tagVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: Recommender.parseFileUser(line)).groupByKey()
        user_meanRatesTags=user_tagVal_pairs.map(lambda p: TagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesTags=spEnv.getSc().broadcast(user_meanRatesTags)
        # print("\nDIZ COM: {}".format(user_meanRatesTags))

        """
        Calcolo delle somiglianze tra users in base a Communities e creazione della corrispondente broadcast variable
        """
        # Ottengo l'RDD (tag,[(user,score),(user,score),...]
        tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: Recommender.parseFileItem(line)).groupByKey().cache()
        user_simsOrd=self.computeSimilarity(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value)
        # usersSimil=spEnv.getSc().broadcast(user_simsOrd)
        # print("\n\nSim Users: {}".format(user_simsOrd.take(2)))

        """
        Calcolo delle (Top_N) raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsOrd.map(lambda p: TagBased.recommendationsUserBased(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        # print("\nLista suggerimenti: {}".format(self.dictRec))


    @staticmethod
    def recommendationsUserBased(user_id,users_with_sim,userHistoryRates,user_meanRates):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        :param user_id: Utente per il quale si calcolano gli items da suggerire
        :param items_with_rating: Lista di items votati dall'utente
        :param user_sims: Matrice di somiglianza tra items calcolata precedentemente
        :param user_meanRates: Dizionario che contiene i rate medi per ogni item
        :return: Pair RDD del tipo: (user,[(scorePred,item),...])
        """
        # Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        # Ciclo su tutti i vicini dell'utente
        for (vicino,(sim,tags)) in users_with_sim:
            # Recupero tutti i rates dati dal tale vicino dell'utente
            listRatings = userHistoryRates.get(vicino,None)
            if listRatings:
                # Ciclo su tutti i Rates
                for (item,rate) in listRatings:
                    # Aggiorno il valore di rate e somiglianza per l'item preso in considerazione
                    totals[item] += sim * (rate-user_meanRates.get(vicino,None))
                    sim_sums[item] += abs(sim)
        # Creo la lista dei rates normalizzati associati agli items per ogni user
        scored_items = [(user_meanRates.get(user_id,None)+(total/sim_sums[item]),item) for item,total in totals.items() if sim_sums[item]!=0.0]
        # Ordino la lista secondo il valore dei rates
        scored_items.sort(reverse=True)
        # Recupero i soli items
        # ranked_items = [x[1] for x in scored_items]
        return user_id,scored_items

