__author__ = 'maury'

from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from numpy import dot
import numpy as np
import operator
import json
import os
import glob

from recommenders.recommender import Recommender
from conf.confItemBased import weightSim,nNeigh, typeSimilarity
from tools.sparkEnvLocal import SparkEnvLocal
from conf.confDirFiles import dirPathInput


class ItemBased(Recommender):
    def __init__(self,name):
        Recommender.__init__(self,name=name)
        self.weightSim=weightSim
        self.nNeigh=nNeigh
        self.typeSimilarity=typeSimilarity


    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemBased
        :param spEnv: SparkContext di riferimento
        :type spEnv: SparkEnvLocal
        :param directory: Directory che contiene insieme di File che rappresentano il TestSet
        :return:
        """
        """
        Calcolo del rate medio per ogni item e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (item,[(user,score),(user,score),...]
        item_user_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: Recommender.parseFileItem(line)).groupByKey()
        item_meanRates=item_user_pair.map(lambda p: ItemBased.computeMean(p[0],p[1])).collectAsMap()
        dictItem_meanRates=spEnv.getSc().broadcast(item_meanRates)

        """
        Calcolo delle somiglianze tra items e creazione della corrispondente broadcast variable
        """
        nNeigh=self.nNeigh
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...] (i soli utenti del train set considerato)
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: Recommender.parseFileUser(line)).groupByKey().cache()
        item_simsOrd=self.computeSimilarity(spEnv,user_item_pair,dictItem_meanRates.value).map(lambda p: ItemBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None).collectAsMap()
        itemsSimil=spEnv.getSc().broadcast(item_simsOrd)
        # print("\n\nSim Items: {}".format(item_simsOrd))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_item_pair.map(lambda p: ItemBased.computeListSugg(p[0],p[1],itemsSimil.value)).mapValues(lambda p: ItemBased.nearestNeighbors(p,nNeigh)).map(lambda p: ItemBased.recommendationsItemsBased(p[0],p[1],dictItem_meanRates.value)).map(lambda p: ItemBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        print("\nLista di raccomandazioni calcolata!")
        print("\nLista suggerimenti: {}".format(self.dictRec))

    def computeSimilarity(self,spEnv,user_item_pair,dictItem_meanRates):
        """
        Vado a calcolare per ogni item la lista di tutti gli items simili in ordine decrescente di somiglianza (pesata sul numero di voti dati da users in comune)
        :param user_item_pair: Pair RDD del tipo (user,[(item,score),(item,score),...]
        :param dictItem_meanRates: Dizionario che per ogni item ha associato il valore medio dei rates
        :return: pair RDD del tipo: (item,[(item,(valSom,[user1,user2,...]),...]
        """
        print("\nInizio del calcolo delle somiglianze tra ITEMS in base ai USERS!")
        pairWiseType="Items"
        # Rimuovo tutti i files Json presenti nella cartella "PairWise" altrimenti la creo
        if not os.path.exists(dirPathInput+"PairWise"+pairWiseType+"/"):
            os.makedirs(dirPathInput+"PairWise"+pairWiseType+"/")
        else:
            # Cancello contenuto cartella
            filelist=glob.glob(dirPathInput+"PairWise"+pairWiseType+"/*")
            for f in filelist:
                os.remove(f)

        # Costruisco un pairRDD del tipo (item1,item2),(user,(rate1,rate2))
        user_item_pair.foreach(lambda p: ItemBased.findItemPairs(p[0],p[1],pairWiseType))

        # Costruisco un pairRDD del tipo (item1,item2),[(user,(rate1,rate2)),(user,(rate1,rate2)),...]
        pairwise_items=spEnv.getSc().textFile(dirPathInput+"PairWise"+pairWiseType+"/*").map(lambda x: json.loads(x)).map(lambda x: (tuple(x[0]),x[1])).groupByKey()

        # Costruisco un pairRDD del tipo (item,[(item1,(val_sim1,{user,user,..})),(item1,(val_sim1,{user,user,..})),...]) ordinato dove i valori di somiglianza sono calcolati in base alla misura scelta e pesati in base al numero di Users in comune
        item_simsWeight=None
        if self.typeSimilarity=="Pearson":
            item_simsWeight = pairwise_items.map(lambda p: ItemBased.pearsonSimilarity(p[0],p[1],dictItem_meanRates)).map(lambda p: ItemBased.keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: ItemBased.computeWeightSim(p[0],p[1],weightSim))
        elif self.typeSimilarity=="Cosine":
            item_simsWeight = pairwise_items.map(lambda p: ItemBased.cosineSimilarity(p[0],p[1])).map(lambda p: ItemBased.keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: ItemBased.computeWeightSim(p[0],p[1],weightSim))
        print("\nFinito di calcolare le somiglianze!")
        return item_simsWeight

    @staticmethod
    def computeListSugg(user_id,items_with_rating,item_sims):
        """
        Per ogni utente ritorno una lista (personalizzata) contenente gli items predetti che faranno parte della lista dei suggerimenti con associata la lista [(itemVotato,sim),(itemVotato,sim),...]
        N.B: Per alcuni user non sarà possibile raccomandare alcun item (Lista vuota) -> Per gli items votati dall'activer la lista delle somiglianze è vuota
        :param user_id: Utente per il quale si calcolano gli items da suggerire
        :param items_with_rating: Lista di items votati dall'utente del tipo [(item,score),(item,score),...]
        :param item_sims: Dizionario di somiglianza tra items calcolata precedentemente item:[(item,(valSom,[user1,user2,...])]
        :return: Pair RDD del tipo: (user,[(itemPredetto,[(itemVotato,rating,sim),(itemVotato,rating,sim),...])
        """
        """ Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori """
        dictSugg= defaultdict(list)
        # Ciclo su tutti i ratings rilasciati dall'active user
        for (item,rating) in items_with_rating:
            # Recupero di tutti i vicini dell'item considerato
            neighbors= item_sims.get(item)
            if neighbors:
                # Per ogni vicino considerato aggiungo l'item a cui assomiglia e corrispondente valore di Somiglianza
                for (neighbor,(sim,_)) in neighbors:
                    dictSugg[neighbor].append((item,rating,sim))

        return user_id,list(dictSugg.items())

    @staticmethod
    def nearestNeighbors(listPredItems,n):
        """
        Ordino la lista dei suggerimenti a seconda dei valori di somiglianza e recupero i primi "n" elementi
        :param listPredItems: Lista dei suggerimenti i cui items (vicini) votati dall'utente devono essere ordinati e filtrati
        :param n: Numero dei primi "n" vicini da prendere in considerazione in ordine di somiglianza
        :return: Lista dei suggerimenti filtrata
        """
        return [(itemPred,sorted(item_rate_sim,key=operator.itemgetter(2),reverse=True)[:n]) for itemPred,item_rate_sim in listPredItems]

    @staticmethod
    def recommendationsItemsBased(user_id,listPredItems,dictItem_meanRates):
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        # Ciclo su tutti i primi N vicini del tale item
        for neighbor,listTriple in listPredItems:
            if listTriple:
                for (item,rate,sim) in listTriple:
                    # Aggiorno il valore di rate e somiglianza per il vicino in questione
                    totals[neighbor] += sim * (rate-dictItem_meanRates.get(item))
                    sim_sums[neighbor] += abs(sim)

        """
            N.B. La lista potrà essere anche vuota se per tutti gli items votati dall'utente non esisterà nemmeno un vicino
            Rilascio la lista dei soli items che ha senso suggerire, quelli più somiglianti complessivamente, tenendo conto di tutti gli items votati dall'utente
        """
        # Creo la lista completa dei rates predetti normalizzati associati ai vicini dei vari item votati dall'utente.
        scored_items = [(dictItem_meanRates.get(neighbor)+(total/sim_sums[neighbor]),neighbor) for neighbor,total in dict(totals).items()]
        # Ordino la lista secondo il valore dei rates
        return user_id,sorted(scored_items,key=operator.itemgetter(0),reverse=True)

    @staticmethod
    def findItemPairs(user,items_with_rating,pairWiseType):
        user=user.replace("/","_")
        with open(dirPathInput+"PairWise"+pairWiseType+"/"+user+".json","w") as f:
            """ Ciclo su tutte le possibili combinazioni di item votati dall'utente restituendone le coppie con relativi rates """
            for item1,item2 in combinations(items_with_rating,2):
                linea=((item1[0],item2[0]),(user,(item1[1],item2[1])))
                f.write(json.dumps(linea)+"\n")

    @staticmethod
    def cosineSimilarity(item_pair,user_rating_pairs):
        """
        Per ogni coppia di items ritorno il valore di somiglianza con annessi numero di voti in comune utilizzando la COSINE SIMILARITY
        N.B: ATTENZIONE! Quando la coppia di item è stata votata da un solo user il valore di somiglianza è sempre di 1 (a prescindere dai rates)
        :param item_pair: Tupla del tipo (item1,item2)
        :param user_rating_pairs: Lista di tuple del tipo [(user,(rate1,rate2)),(user,(rate1,rate2)),...]
        :return: Elemento del tipo (item1,item2),(valSim,{User1,User2,...})
        """
        users,rating_pairs=zip(*user_rating_pairs)
        item1Rates,item2Rates=zip(*rating_pairs)
        item1Rates=np.array(item1Rates)
        item2Rates=np.array(item2Rates)
        cos_sim=1-cosine(item1Rates,item2Rates)
        return item_pair,(cos_sim,list(set(users)))

    @staticmethod
    def pearsonSimilarity(item_pair,user_rating_pairs,item_meanRates):
        """
        Calcola la somiglianza tra items utilizzando la Person come misura quando possibile (den!=0) o la cosine in caso contrario
        :param item_pair: Tupla del tipo (item1,item2)
        :param user_rating_pairs: Lista di tuple del tipo [(user,(rate1,rate2)),(user,(rate1,rate2)),...]
        :param item_meanRates: Dizionario che contiene i rate medi per ogni item
        :return: Elemento del tipo (item1,item2),(valSim,{User1,User2,...})
        """
        users,rating_pairs=zip(*user_rating_pairs)
        item1Rates,item2Rates=zip(*rating_pairs)
        item1Rates=np.array(item1Rates)
        item2Rates=np.array(item2Rates)
        item1=item_pair[0]
        item2=item_pair[1]
        den=norm((item1Rates-item_meanRates[item1]))*norm((item2Rates-item_meanRates[item2]))
        if den!=0.0:
            cos_sim=dot((item1Rates-item_meanRates.get(item1,None)),(item2Rates-item_meanRates.get(item2,None)))/den
        else:
            cos_sim=1-cosine(item1Rates,item2Rates)
        return item_pair,(cos_sim,list(set(users)))

    @staticmethod
    def computeWeightSim(item_id,items_and_sims,weightSim):

        def computeWeightSimElem(weight,elem):
            return (elem[0], ((len(elem[1][1])/weight)*elem[1][0],elem[1][1])) if len(elem[1][1])<weight else elem

        items_and_simsWeight=[computeWeightSimElem(weight=weightSim,elem=el) for el in items_and_sims]
        return item_id,items_and_simsWeight

    @staticmethod
    def keyOnFirstItem(item_pair,item_sim_data):
        """ Per ogni coppia di items faccio diventare il primo la key """
        (item1_id,item2_id) = item_pair
        return item1_id,(item2_id,item_sim_data)

    @staticmethod
    def filterSimilarities(item_id,items_and_sims):
        """
        Rimuovo tutti quei items per i quali il valore di somiglianza è < 0.5
        :param item_id: Item preso in considerazione
        :param items_and_sims: Items e associati valori di somiglianze per l'item sotto osservazione
        :return: Ritorno un nuovo pairRDD filtrato
        """
        lista=[item for item in items_and_sims if item[1][0]>=0.5]
        if len(lista)>0:
            return item_id,lista

    @staticmethod
    def convertFloat_Int(user_id,recommendations):
        """
        Converto da float a int il valore predetto per ogni elemento
        :param user_id: utente per il quale si vengono suggeriti gli items
        :param recommendations: lista di items raccomandati con associato valore (float)
        :return: user_id,recommendations
        """
        lista=[(int(pred),item) for pred,item in recommendations]
        return user_id,lista




