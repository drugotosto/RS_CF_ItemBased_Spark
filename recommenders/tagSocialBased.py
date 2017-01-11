__author__ = 'maury'

import json

from recommenders.tagBased import TagBased
from recommenders.socialBased import SocialBased
from recommenders.itemBased import ItemBased
from conf.confCommunitiesFriends import fileFriendsCommunities
from conf.confDirFiles import userTagJSON


class TagSocialBased(TagBased,SocialBased):
    def __init__(self,name,friendships):
        ItemBased.__init__(self,name=name)
        self.friendships=friendships

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemSocialBased
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: TagSocialBased.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: TagSocialBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo media dei valori associati ai Tags (per ogni user) e creazione della corrispondente broadcast variable (Utilizzo per misura Pearson)
        """
        user_tagVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagSocialBased.parseFileUser(line)).groupByKey()
        user_meanRatesTags=user_tagVal_pairs.map(lambda p: TagSocialBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesTags=spEnv.getSc().broadcast(user_meanRatesTags)
        # print("\nDIZ COM: {}".format(user_meanRatesTags))

        """
        Calcolo delle somiglianze tra users in base a Tags e creazione del corrispondente RDD
        """
        tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagSocialBased.parseFileItem(line)).groupByKey().cache()
        # Ottengo l'RDD (user,[(user,(ValSim,[Tag,Tag,...])),(user,(ValSim,[Tag,Tag,...])),...])
        user_simsTags=self.computeSimilarity(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value).map(lambda p: TagBased.removeOneRate(p[0],p[1])).filter(lambda p: p!=None)
        print("\nFinito di calcolare RDD Somiglianze tra Users in base a TAGS!")
        for user,user_valPairs in user_simsTags.take(1):
            print("\nUserUser : {}".format(user))
            print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle somiglianze tra users in base alle CommunitiesFriends e creazione del corrispondente RDD
        """
        comm_listUsers=spEnv.getSc().textFile(fileFriendsCommunities).map(lambda x: json.loads(x))
        # Ottengo l'RDD (user,[(user,ValSim),(user,ValSim),...])
        user_simsOrdFriends=self.computeSimilarityFriends(spEnv,comm_listUsers).filter(lambda p: p!=None)
        print("\nFinito di calcolare RDD Somiglianze tra Users in base a FRIENDS!")
        print(user_simsOrdFriends.take(1))
        for user,user_valPairs in user_simsOrdFriends.take(1):
            print("\nUser : {}".format(user))
            print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Creazione RDD delle somiglianze finale che tiene conto dei due RDD (Variante 1) (Senza filtro Neighboors su TAGS)
        """
        # Modifica dell'RDD relativo ai TAGS
        user_simsTags=user_simsTags.mapValues(lambda x: TagSocialBased.RemoveTags(x))
        # print("\nSemplificato RDD_TAGS Somiglianze!")
        nNeigh=self.nNeigh
        user_simsTot=user_simsTags.union(user_simsOrdFriends).reduceByKey(lambda listPair1,listPair2: TagSocialBased.joinPairs(listPair1,listPair2)).map(lambda p: TagSocialBased.nearestNeighbors(p[0],p[1],nNeigh)).cache()
        print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(1)))

        """
        Creazione RDD delle somiglianze finale che tiene conto dei due RDD (Variante 2) (Con filtro Neighboors su TAGS)
        """
        # Modifica dell'RDD relativo ai TAGS
        user_simsTags=user_simsTags.mapValues(lambda x: TagSocialBased.RemoveTags(x))
        nNeigh=self.nNeigh
        user_simsTagsN=user_simsTags.map(lambda p: TagSocialBased.nearestNeighbors(p[0],p[1],nNeigh)).collectAsMap()
        dictUser_simsTags=spEnv.getSc().broadcast(user_simsTagsN)
        user_simsTot=user_simsOrdFriends.map(lambda p: TagSocialBased.unisco(p[0],p[1],dictUser_simsTags.value)).cache()
        print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(1)))

        """
        Calcolo delle (Top_N) raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsTot.map(lambda p: TagSocialBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    @staticmethod
    def RemoveTags(listPairs):
        return [(user,pair[0]) for user,pair in listPairs if pair[0]>0.0]

    @staticmethod
    def joinPairs(listPair1,listPair2):
        return [(pair1[0],pair1[1]+pair2[1]) for pair1 in listPair1 for pair2 in listPair2 if pair1[0]==pair2[0]]

    @staticmethod
    def unisco(user_id,users_with_sim,dictUser_simsTags):
        # Controllo la presenza dell'utente all'interno del dizionario
        if user_id in dictUser_simsTags:
            dictUsersTagSim=dict(dictUser_simsTags[user_id])
            # Ciclo sulle coppie di (utente,valSim) e costruisco la lista delle somiglianze finale
            lista=[(userFriendSim,valFriendSim+dictUsersTagSim[userFriendSim]) for userFriendSim,valFriendSim in users_with_sim if userFriendSim in dictUsersTagSim.keys()]
            return user_id,lista


