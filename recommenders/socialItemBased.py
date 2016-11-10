__author__ = 'maury'

from collections import defaultdict

from recommenders.itemBased import ItemBased

class SocialItemBased(ItemBased):
    def __init__(self,name,friends):
        ItemBased.__init__(self,name=name)
        self.friends=friends


    def recommendations(self,user_id,items_with_rating,item_sims,item_meanRates):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        :param user_id: Utente per il quale si calcolano gli items da suggerire
        :param items_with_rating: Lista di items votati dall'utente
        :param item_sims: Matrice di somiglianza tra items calcolata precedentemente
        :param item_meanRates: Dizionario che contiene i rate medi per ogni item
        :return: Lista di items suggeriti per ogni user preso in considerazione
        """
        print("\nRec Type: {}".format(self.name))
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
                        totals[neighbor] += sim * (rating-item_meanRates.get(neighbor,None))
                        sim_sums[neighbor] += abs(sim)
        # Creo la lista dei rates normalizzati associati agli items per ogni user
        scored_items = [(item_meanRates.get(item,None)+(total/sim_sums[item]),item) for item,total in totals.items() if sim_sums[item]!=0.0]
        # Ordino la lista secondo il valore dei rates
        scored_items.sort(reverse=True)
        # Recupero i soli items
        # ranked_items = [x[1] for x in scored_items]
        return user_id,scored_items
