import os
from conf.confItemBased import weightSim
from recommenders.tagBased import TagBased

__author__ = 'maury'

import json

from recommenders.socialBased import SocialBased
from recommenders.itemBased import ItemBased
from conf.confCommunitiesFriends import fileFriendsCommunities
from conf.confDirFiles import userTagJSON, dirPathInput, dirPathCommunities


class TagSocialBased(SocialBased,TagBased):
    def __init__(self,name,friendships,communityType):
        SocialBased.__init__(self,name,friendships,communityType)

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
        Calcolo delle somiglianze tra users in base ai TAGS e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/"):
            print("\nNon esistono ancora i valori di somiglianza tra Users. Vado a calcolarli!")
            nNeigh=self.nNeigh
            # Ottengo l'RDD con elementi (tag,[(user1,score),(user2,score),...]
            tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileItem(line)).groupByKey().cache()
            user_simsTags=self.computeSimilarityTag(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value).map(lambda p: TagBased.nearestNeighbors(p[0],p[1],nNeigh)).map(lambda p: TagBased.removeOneRate(p[0],p[1])).filter(lambda p: p!=None)
            user_simsTags.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/")
        else:
            print("\nLa somiglianza tra Users e già presente")
            user_simsTags=spEnv.getSc().textFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/").map(lambda x: json.loads(x))

        # for user,user_valPairs in user_simsTags.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle somiglianze tra users in base alle CommunitiesFriends e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/"):
            nNeigh=self.nNeigh
            comm_listUsers=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json").map(lambda x: json.loads(x))
            user_simsFriends=self.computeSimilarityFriends(spEnv,comm_listUsers).map(lambda p: SocialBased.nearestNeighbors(p[0],p[1],nNeigh)).filter(lambda p: p!=None)
            user_simsFriends.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/")
        else:
            print("\nLe somiglianze tra friends appartenenti alle stesse communities (trovate dall'algoritmo {}) sono già presenti!".format(self.communityType))
            user_simsFriends=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/").map(lambda x: json.loads(x))

        # for user,user_valPairs in user_simsFriends.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))


        """
        Creazione RDD delle somiglianze finale che tiene conto dei due RDD (Variante 1) (Senza filtro Neighboors su TAGS)
        """
        # Modifica dell'RDD relativo ai TAGS
        user_simsTags=user_simsTags.mapValues(lambda x: TagSocialBased.RemoveTags(x))
        # print("\nSemplificato RDD_TAGS Somiglianze!")
        nNeigh=self.nNeigh
        user_simsTot=user_simsTags.union(user_simsFriends).reduceByKey(lambda listPair1,listPair2: TagSocialBased.joinPairs(listPair1,listPair2)).map(lambda p: TagSocialBased.nearestNeighbors(p[0],p[1],nNeigh)).cache()
        print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(1)))

        """
        # Creazione RDD delle somiglianze finale che tiene conto dei due RDD (Variante 2) (Con filtro Neighboors su TAGS)
        # """
        # # Modifica dell'RDD relativo ai TAGS
        # user_simsTags=user_simsTags.mapValues(lambda x: TagSocialBased.RemoveTags(x))
        # nNeigh=self.nNeigh
        # user_simsTagsN=user_simsTags.map(lambda p: TagSocialBased.nearestNeighbors(p[0],p[1],nNeigh)).collectAsMap()
        # dictUser_simsTags=spEnv.getSc().broadcast(user_simsTagsN)
        # user_simsTot=user_simsFriends.map(lambda p: TagSocialBased.unisco(p[0],p[1],dictUser_simsTags.value)).cache()
        # print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        # print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(1)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsTot.map(lambda p: TagSocialBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagSocialBased.convertFloat_Int(p[0],p[1])).collectAsMap()
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
        else:
            lista=[(userFriendSim,valFriendSim) for userFriendSim,valFriendSim in users_with_sim]

        return user_id,lista


