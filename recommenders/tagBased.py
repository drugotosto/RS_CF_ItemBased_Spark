__author__ = 'maury'

import json
import os
from collections import defaultdict

from conf.confDirFiles import userTagJSON, dirPathInput
from recommenders.itemBased import ItemBased
from tools.sparkEnvLocal import SparkEnvLocal
from conf.confItemBased import weightSim

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
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: TagBased.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: TagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo media dei valori associati ai TAGS (per ogni user) e creazione della corrispondente broadcast variable (Utilizzo per misura Pearson)
        """
        # Ottengo RDD con elementi (user,[(tag1,score),(tag2,score),...]
        user_tagVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileUser(line)).groupByKey()
        user_meanRatesTags=user_tagVal_pairs.map(lambda p: TagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesTags=spEnv.getSc().broadcast(user_meanRatesTags)

        """
        Calcolo delle somiglianze tra users in base ai TAGS e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/"):
            print("\nNon esistono ancora i valori di somiglianza tra Users. Vado a calcolarli!")
            nNeigh=self.nNeigh
            # Ottengo l'RDD con elementi (tag,[(user1,score),(user2,score),...]
            tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileItem(line)).groupByKey().cache()
            user_simsOrd=self.computeSimilarityTag(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value).map(lambda p: TagBased.nearestNeighbors(p[0],p[1],nNeigh)).map(lambda p: TagBased.removeOneRate(p[0],p[1])).filter(lambda p: p!=None)
            user_simsOrd.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/")
        else:
            print("\nLa somiglianza tra Users e già presente")
            user_simsOrd=spEnv.getSc().textFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/").map(lambda x: json.loads(x))

        # for user,user_valPairs in user_simsOrd.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        # Recupero storico dei Ratings dei vari utenti e ne faccio una V.B.
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsOrd.map(lambda p: TagBased.recommendationsUserBasedTag(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        print("\nLista di raccomandazioni calcolata!")
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    def computeSimilarityTag(self,spEnv,tag_userVal_pairs,dictUser_meanRatesTags):
        """
        Vado a calcolare per ogni user la lista dei Top-N users più simili in ordine decrescente di somiglianza (pesata sul numero di tags in comune)
        :param tag_userVal_pairs: Pair RDD del tipo: (tag,[(user1,score),(user2,score),...]
        :param dictUser_meanRatesTags: Dizionario che per ogni item ha associato il valore medio dei valori associato ai tags
        :return: pair RDD del tipo: (user,[(user1,(valSom,[tag1,tag2,...]),...]
        """
        print("\nInizio del calcolo delle somiglianze tra USERS in base ai TAGS!")
        pairWiseType="Users"
        if not os.path.exists(dirPathInput+"PairWise"+pairWiseType+"/"):
            print("\nCreazione PairWise Users")
            os.makedirs(dirPathInput+"PairWise"+pairWiseType+"/")
            # Costruisco un pairRDD del tipo (user1,user2),(tag,(rate1,rate2))
            tag_userVal_pairs.foreach(lambda p: ItemBased.findItemPairs(p[0],p[1],pairWiseType))
        else:
            print("\nPairWise Users già presente")

        # Costruisco un pairRDD del tipo (user1,user2),[(tag,(rate1,rate2)),(tag,(rate1,rate2)),...]
        pairwise_items=spEnv.getSc().textFile(dirPathInput+"PairWise"+pairWiseType+"/*").map(lambda x: json.loads(x)).map(lambda x: (tuple(x[0]),x[1])).groupByKey()

        """ Codice per calcolare il 'miglior' valore di WeightSim """
        # print("\nCalcolo del valore medio di TAGS/USERS condivisi tra USERS/ITEMS")
        # numShare=pairwise_items.map(lambda p: TagBased.getNumbShareTags(p[0],p[1])).reduce(lambda x, y: x + y)
        # weightSim=numShare/pairwise_items.count()
        # if "Tag" in self.getName():
        #     print("\nnumTagsShare: {}".format(numShare))
        #     print("Numero medio Tags condivisi tra coppie di utenti: {}".format(numShare/pairwise_items.count()))

        # Costruisco un pairRDD del tipo (user,[(user1,(val_sim1,{tag1,tag2,..})),(user2,(val_sim2,{tag1,tag2,..})),...]) ordinato e  con valori pesati in base al numero di TAGS in comune
        user_simsWeight=None
        if self.typeSimilarity=="Pearson":
            user_simsWeight = pairwise_items.map(lambda p: TagBased.pearsonSimilarity(p[0],p[1],dictUser_meanRatesTags)).map(lambda p: TagBased.keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: TagBased.computeWeightSim(p[0],p[1],weightSim))
        elif self.typeSimilarity=="Cosine":
            user_simsWeight = pairwise_items.map(lambda p: TagBased.cosineSimilarity(p[0],p[1])).map(lambda p: TagBased.keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: TagBased.computeWeightSim(p[0],p[1],weightSim))
        print("\nFinito di calcolare le somiglianze!")
        return user_simsWeight

    @staticmethod
    def recommendationsUserBasedTag(user_id,users_with_sim,userHistoryRates,user_meanRates):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        :param user_id: Utente per il quale si calcolano gli items da suggerire
        :param users_with_sim: Lista di items votati dall'utente
        :param userHistoryRates: Matrice di somiglianza tra items calcolata precedentemente
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
        """
            N.B. La lista potrà essere anche vuota se per tutti gli items votati dall'utente non esisterà nemmeno un vicino con valore di somiglianza complessivo > 0.0
            Rilascio la lista dei soli items che ha senso suggerire, quelli più somiglianti complessivamente, tenendo conto di tutti gli items votati dall'utente
        """
        scored_items = [(user_meanRates.get(user_id,None)+(total/sim_sums[item]),item) for item,total in totals.items() if sim_sums[item]>0.0]
        # Ordino la lista secondo il valore dei rates
        scored_items.sort(reverse=True)
        # Recupero i soli items
        # ranked_items = [x[1] for x in scored_items]
        return user_id,scored_items

    @staticmethod
    def getNumbShareTags(item_pair,user_rating_pairs):
        users,_=zip(*user_rating_pairs)
        return len(users)

    # def SaveSimilarities(user_id,listUsers):
