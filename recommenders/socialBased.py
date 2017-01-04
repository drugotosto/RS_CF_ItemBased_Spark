__author__ = 'maury'

from itertools import combinations
from os.path import isfile
from igraph import *
import json
import time

from tools.sparkEnvLocal import SparkEnvLocal
from recommenders.itemBased import ItemBased
from recommenders.recommender import Recommender
from conf.confDirFiles import userFriendsGraph
from conf.confCommunitiesFriends import *
from tools.tools import saveJsonData

class SocialBased(ItemBased):
    def __init__(self,name,friendships):
        ItemBased.__init__(self,name=name)
        # Settaggio del dizionazio delle amicizie (inzialmente non pesato)
        self.friendships=friendships

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
        user_meanRatesRatings=user_item_pair.map(lambda p: SocialBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo delle somiglianze tra users in base agli amici in comune e creazione della corrispondente broadcast variable
        """
        if not os.path.exists(dirPathCommunities+communityType+"/SimilaritiesFiles"):
            os.makedirs(dirPathCommunities+communityType+"/SimilaritiesFiles")
        comm_listUsers=spEnv.getSc().textFile(fileFriendsCommunities).map(lambda x: json.loads(x))
        friendships=spEnv.getSc().broadcast(self.friendships)
        user_simsOrd=comm_listUsers.foreach(lambda x: SocialBased.computeCommunitiesSimilarity(x[0],x[1],friendships.value))

        # """
        # Calcolo delle (Top_N) raccomandazioni personalizzate per i diversi utenti
        # """
        # user_item_hist=user_item_pair.collectAsMap()
        # userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        # user_item_recs = user_simsOrd.map(lambda p: TagBased.recommendationsUserBased(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        # self.setDictRec(user_item_recs)
        # # print("\nLista suggerimenti: {}".format(self.dictRec))

    @staticmethod
    def computeCommunitiesSimilarity(comm,listUsers,friendships):
        with open(dirPathCommunities+communityType+"/SimilaritiesFiles/communitiesFriendsSim_"+str(comm)+".json","w") as f:
            """ Ciclo su tutte le possibili combinazioni di users che appartengono alla stessa community """
            for user1,user2 in combinations(listUsers,2):
                amici_user1=list(zip(*friendships[user1]))[0]
                amici_user2=list(zip(*friendships[user2]))[0]
                num=len(set(amici_user1+tuple(user1)).intersection(set(amici_user2+tuple(user2))))+1
                den=len(set(amici_user1+tuple(user1)).union(set(amici_user2+tuple(user2))))+1
                linea=((user1,(user2,num/den)))
                f.write(json.dumps(linea)+"\n")
                linea=((user2,(user1,num/den)))
                f.write(json.dumps(linea)+"\n")

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
        pass

    def createFriendsCommunities(self):

        # Creazione del dizionario delle amicizie
        self.createDizFriendships()

        # Controllo esistenza del Grafo delle amicizie e nel caso lo vado a creare
        if not isfile(userFriendsGraph):
            self.createGraph()

        # Calcolo le communities delle amicizie
        self.createCommunities()

    def createDizFriendships(self):
        def createPairs(user,listFriends):
            return [(user,friend) for friend in listFriends]

        """ Creo gli archi del grafo mancanti """
        listaList=[createPairs(user,listFriends) for user,listFriends in self.friendships.items()]
        archiPresenti={coppia for lista in listaList for coppia in lista}
        archiMancanti={(arco[1],arco[0]) for arco in archiPresenti if (arco[1],arco[0]) not in archiPresenti}
        # print("\n- Numero di archi mancanti: {}".format(len(archiMancanti)))
        archiDoppi=archiPresenti.union(archiMancanti)
        # print("\n- Numero di archi/Amicizie (doppie) totali presenti sono: {}".format(len(archiDoppi)))

        """ Costruisco il dizionario con archi doppi senza peso sugli archi """
        dizFriendshipsDouble=defaultdict(list)
        for k, v in archiDoppi:
            dizFriendshipsDouble[k].append(v)
        # print("\n- Numero di utenti: {}".format(len([user for user in dizFriendshipsDouble])))

        """ Costruisco il dizionario con gli archi pesati (dato dal numero di amicizie in comune tra utenti) """
        def createListFriendsDoubleWeight(user,dizFriendshipsDouble):
            return [(friend,len(set(dizFriendshipsDouble[user])&set(dizFriendshipsDouble[friend]))+1) for friend in dizFriendshipsDouble[user]]

        friendships={user:createListFriendsDoubleWeight(user,dizFriendshipsDouble) for user in dizFriendshipsDouble}

        # """ Per ogni arco (user1-user2) vado ad eliminare la controparte (user2-user1) """
        # archi=set()
        # for user,listFriends in dizFriendshipsDoubleWeight.items():
        #     for elem in listFriends:
        #         if (user,elem[0],elem[1]) not in archi and (elem[0],user,elem[1]) not in archi:
        #             archi.add((user,elem[0],elem[1]))
        #
        # """ Costruisco il dizionario finale da salvare """
        # friendships=defaultdict(list)
        # for k,v,r in archi:
        #     friendships[k].append((v,r))

        print("\nNumero di AMICIZIE (doppie) presenti sono: {}".format(sum([len(lista) for lista in friendships.values()])))
        # numUtenti=len(set([user for user in friendships]).union(set([user for lista in friendships.values() for user,_ in lista])))
        numUtenti=len(list(friendships.keys()))
        print("\nNumero di UTENTI che sono presenti in communities: {}".format(numUtenti))
        time.sleep(10)
        self.setFriendships(friendships)
        print("\nDizionario delle amicizie pesato creato e settato!")

    def createGraph(self):
        def saveGraphs(g):
            g.write_pickle(fname=open(userFriendsGraph,"wb"))
            g.write_graphml(f=open(userFriendsGraph+".graphml","wb"))

        g=Graph()
        # Recupero i vertici (users) del grafo delle amicizie
        # users={user for user in self.getFriendships().keys()}.union({friend for listFriends in self.getFriendships().values() for friend,weight in listFriends})
        users={user for user in self.getFriendships().keys()}
        for user in users:
            g.add_vertex(name=user,gender="user",label=user)

        def createPairs(user,listItems):
            return [(user,item[0]) for item in listItems]

        # Creo gli archi (amicizie) del grafo singole
        listaList=[createPairs(user,listItems) for user,listItems in self.getFriendships().items()]
        archi=[]
        for lista in listaList:
            for user,elem in lista:
                if (elem,user) not in archi:
                    archi.append((user,elem))
        g.add_edges(archi)
        # Aggiungo i relativi pesi agli archi
        weights=[weight for user,listItems in self.getFriendships().items() for _,weight in listItems]
        g.es["weight"]=weights

        saveGraphs(g)
        print("\nSummary:\n{}".format(summary(g)))
        print("\nGrafo delle amicizie creato e salvato!")

    def createCommunities(self):
        startTime=time.time()
        g=Graph.Read_Pickle(fname=open(userFriendsGraph,"rb"))
        # calculate dendrogram
        dendrogram=None
        clusters=None
        if communityType=="all":
            types=communitiesTypes
        else:
            types=[communityType]

        for type in types:
            if type=="fastgreedy":
                dendrogram=g.community_fastgreedy(weights="weight")
            elif type=="walktrap":
                dendrogram=g.community_walktrap(weights="weight")
            elif type=="label_propagation":
                clusters=g.community_label_propagation(weights="weight")
            elif type=="multilevel":
                clusters=g.community_multilevel(weights="weight",return_levels=False)
            elif type=="infomap":
                clusters=g.community_infomap(edge_weights="weight")

            # convert it into a flat clustering (VertexClustering)
            if type!="label_propagation" and type!="multilevel" and type!="infomap":
                clusters = dendrogram.as_clustering()
            # get the membership vector
            membership = clusters.membership
            communitiesFriends=defaultdict(list)
            for user,community in [(name,membership) for name, membership in zip(g.vs["name"], membership)]:
                communitiesFriends[community].append(user)
            saveJsonData(communitiesFriends.items(),dirPathCommunities+"/"+type,dirPathCommunities+"/"+type+"/communitiesFriends.json")
            print("\nClustering Summary for '{}' : \n{}".format(type,clusters.summary()))

        print("\nFinito di calcolare le communities in TEMPO: {}".format((time.time()-startTime)/60))

    def setFriendships(self,friendships):
        self.friendships=friendships

    def getFriendships(self):
        return self.friendships



