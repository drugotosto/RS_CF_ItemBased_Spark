__author__ = 'maury'

from collections import defaultdict

from recommenders.itemBased import ItemBased
from recommenders.recommender import Recommender

class TagSocialBased(ItemBased):
    def __init__(self,name,friendships):
        ItemBased.__init__(self,name=name)
        self.friendships=friendships

    def builtModel(self,sc,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemSocialBased
        :return:
        """
        """
        Calcolo del rate medio per ogni item e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (item,[(user,score),(user,score),...]
        item_user_pair=sc.textFile(directory+"/*").map(lambda line: Recommender.parseFileItem(line)).groupByKey()
        item_meanRates=item_user_pair.map(lambda p: ItemBased.computeMean(p[0],p[1])).collectAsMap()
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
        friends=sc.broadcast(self.getFriends())
        user_item_recs = user_item_pair.map(lambda p: SocialBased.recommendations(p[0],p[1],itemsSimil.value,dictItem_meanRates.value,friends.value)).map(lambda p: ItemBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)

        ris=[(user,listSugg) for user,listSugg in self.dictRec.items() if listSugg]
        if ris:
            print(ris)
        else:
            print("VUOTA")

    @staticmethod
    def recommendations(user_id,items_with_rating,item_sims,item_meanRates,friends):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        Versione che tiene conto solamente dei valori di somiglianza derivanti dagli amici dell'active user
        :param user_id:
        :param items_with_rating:
        :param item_sims:
        :param item_meanRates:
        :param friends:
        :return:
        """
        # Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        for (item,rating) in items_with_rating:
            # Recupero tutti i vicini del tale item
            nearest_neighbors = item_sims.get(item,None)
            if nearest_neighbors:
                # Ciclo su tutti i vicini del tale item
                for (neighbor,(sim,setUsers)) in nearest_neighbors:
                    if onlyFriends:
                        print("\nAMICI: {}".format(friends[user_id]))
                        if neighbor!=item and friends[user_id] and friends[user_id]==setUsers:
                            # Aggiorno il valore di rate e somiglianza per il vicino in questione
                            totals[neighbor] += sim * (rating-item_meanRates.get(neighbor,None))
                            sim_sums[neighbor] += abs(sim)
                    else:
                        if neighbor!=item:
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


    def createFriendsCommunities(self):
        pass

    def getFriendships(self):
        return self.friendships
