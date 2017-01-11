import operator

__author__ = 'maury'

from collections import defaultdict, Counter
from pandas import DataFrame
from itertools import islice
import pandas as pd
import json

from conf.confDirFiles import reviewsJSON, businessJSON, usersJSON, dirPathInput, userTagJSON
from conf.confRS import tag, tagToFilter, numTags, numRec
from tools.tools import saveJsonData

class DataScienceAnalyzer():
    def __init__(self):
        self.dataFrame=None
        self.dictBusCat=None
        self.numUsers=None
        self.numBusiness=None
        self.numCatBus=None

    def createDataSet(self,numTags):
        print("\n******** Creo il DataSet di partenza! *********")
        # LETTURA DEL FILE JSON DELLE REVIEWS A PEZZI
        dfRatings=self.createRatingsDF()
        # LETTURA DEL FILE JSON DEI BUSINESS A PEZZI
        dfBusiness=self.createBusinessDF()
        dfUsers=self.createUsersDF()
        # Merge dei dataframe tra RATINGS e BUSINESS
        dfMerge=pd.merge(dfRatings,dfBusiness,left_on='business_id',right_index=True, how="inner")
        # Merge dei dataframe tra RATINGS e BUSINESS e USERS
        dfMerge=pd.merge(dfMerge,dfUsers,left_on='user_id',right_index=True, how="inner")

        print("\nFiltraggio su dfMerge dei soli ratings appartenenti ad una specifica tag scelta: {}".format(tag))
        dfMerge=dfMerge[dfMerge["categories"].apply(lambda x: bool(set(x).intersection([tag])))]

        print("\nFiltraggio su dfMerge degli users che hanno rilasciato un numero di recensioni < "+str(numRec))
        dfMerge=self.userFilterByNumRatings(dfMerge)

        print("\nFiltraggio su dfMerge degli users per i quali tra tutti i tags associati ai business votati neanche 1 tra questi è presente almeno {} volte".format(numTags))
        dfMerge,dizUserTag=self.userFilterByNumTags(dfMerge)

        print("\nCreazione del File Json 'dizUserTag' che associa ad ogni utente la lista dei Tags dei business da lui votati con relativo peso con valori [0-10]")
        self.createUserTagJSON(dizUserTag)

        print("\nPer ogni utente rimuovo tutti i corrispondenti amici che non risultano far parte tra gli utenti finali del Dataset preso in considerazione")
        dfMerge=self.userFilterByFriends(dfMerge)

        print("\nAlla fine di tutto dfMerge potra contenere Users con il campo 'friends' vuoto e/o aver votato un business che non ha associato nessun tag!")

        # Vado a settare i parametri del DataFrame che riguardano: Numero di Business, Numero di Users, Numero di Categorie di Business
        self.setProperties(dfMerge)

        # Stampa valori Dataset Finale
        self.printValuesDataset()

    def createRatingsDF(self):
        utenti=[]
        business=[]
        stars=[]
        with open(reviewsJSON, 'r') as f:
            while True:
                # Vado a gestire una lista di 1000 stringe JSON alla volta
                lines_gen = list(islice(f, 1000))
                if lines_gen:
                    # Vado a gestire una stringa JSON alla volta
                    for line in lines_gen:
                        oggettoJSON=json.loads(line)
                        utenti.append(oggettoJSON["user_id"])
                        business.append(oggettoJSON["business_id"])
                        stars.append(oggettoJSON["stars"])
                else:
                    break

        dfRatings=DataFrame({"user_id": utenti,"business_id": business,"stars": stars},columns=["user_id","business_id","stars"])
        print("\nDataFrame dei Ratings creato!")
        return dfRatings

    def createBusinessDF(self):
        businessID=[]
        businessName=[]
        businessCat=[]
        with open(businessJSON, 'r') as f:
            while True:
                lines_gen = list(islice(f, 1000))
                if lines_gen:
                    for line in lines_gen:
                        oggettoJSON=json.loads(line)
                        businessID.append(oggettoJSON["business_id"])
                        businessName.append(oggettoJSON["name"])
                        businessCat.append(oggettoJSON["categories"])
                else:
                    break

        dfBusiness=DataFrame({"business_name": businessName,"categories": businessCat},columns=["business_name","categories"],index=businessID)
        print("\nDataFrame dei Business creato!")
        return dfBusiness

    def createUsersDF(self):
        usersID=[]
        usersName=[]
        listFriends=[]
        with open(usersJSON, 'r') as f:
            while True:
                lines_gen = list(islice(f, 1000))
                if lines_gen:
                    for line in lines_gen:
                        oggettoJSON=json.loads(line)
                        usersID.append(oggettoJSON["user_id"])
                        usersName.append(oggettoJSON["name"])
                        listFriends.append(oggettoJSON["friends"])
                else:
                    break

        dfUsers=DataFrame({"users_name": usersName,"friends": listFriends},columns=["users_name","friends"],index=usersID)
        print("\nDataFrame degli Users creato!")
        return dfUsers

    def userFilterByNumRatings(self, dfMerge):
        return dfMerge.groupby("user_id").filter(lambda x: len(x) >= numRec)
        # # Recupero lista dei soli users che hanno votato almeno numRec business
        # usersFilt=[user for user,count in dict(dfMerge.groupby("user_id").size()).items() if count>=numRec]
        # # Filtro il dataSet iniziale mantenendo solo gli users recuperati precedentemente
        # return dfMerge[dfMerge["user_id"].isin(usersFilt)]

    def userFilterByNumTags(self, dfMerge):
        # Categorie da rimuovere
        tagsToRemove=set(tagToFilter+[tag])
        print("Insieme dei tags da rimuovere: {}".format(tagsToRemove))

        # Funzione per sostituire elementi della colonna "categories"
        def substitute(listTags,listTagsRemove):
            return [x for x in listTags if x not in listTagsRemove]

        dfMerge["categories"]=dfMerge["categories"].apply(lambda x: substitute(x,tagsToRemove))
        # Creo il dizionario che contiene per ogni utente la lista di tutti i tags (con ripetizione) associate ai business votati dall'utente
        dizUserTag=defaultdict(list)
        for k, v in list(zip(dfMerge["user_id"],dfMerge["categories"])):
            if v:
                dizUserTag[k].extend(v)

        def filterUsers(dizTags):
            for val in dict(dizTags).values():
                if val>=numTags:
                    return [(k,v) for k,v in dict(dizTags).items()]
            return []

        dizUserTag={k:Counter(v) for k,v in dizUserTag.items()}
        dizUserTag={k:filterUsers(v) for k,v in dizUserTag.items()}

        # Rimuovo utenti che non hanno associato nessun Tag
        dizUserTag={user:listPairs for user,listPairs in dizUserTag.items() if len(listPairs)>0}

        return (dfMerge[dfMerge["user_id"].isin(dizUserTag.keys())],dizUserTag)


    def createUserTagJSON(self,dizUserTag):
        # Calcolo del dizionario che associa ad ogni Tag la relativa frequenza all'interna del DataSet
        tagsCounter=Counter([pair[0] for listPairs in dizUserTag.values() for pair in listPairs])
        numTotTags=sum(tagsCounter.values())
        dizTags={tag:1-(val/numTotTags) for tag,val in tagsCounter.items()}

        # Creazione file Json "userTag.json" che associa ogni utente l'insieme dei Tags dei business da lui votati con relativo peso
        saveJsonData([(user,tag,((val/sum(list(zip(*listPairs))[1]))*dizTags[tag])*10) for user,listPairs in dizUserTag.items() for tag,val in listPairs],dirPathInput,userTagJSON)


    def userFilterByFriends(self, dfMerge):
        def filterFriends(x):
            return [item for item in x if item in users]

        users=set(dfMerge["user_id"].values)
        dfMerge["friends"]=dfMerge["friends"].apply(filterFriends)

        return dfMerge

    def getCatBusiness(self, listBusiness):
        """
        Recupero tutti le categorie differenti associate ai vari business
        :return: Numero di categorie differenti
        """
        listCategories=[self.dataFrame[self.dataFrame["business_id"]==business].head(1).categories.values[0] for business in listBusiness]
        numCategories=len(set().union(*listCategories))
        return numCategories

    def retrieveFriends(self):
        """
        Recupero dei diversi friends per ogni utente che fa parte del Dataset
        :return: Dizionario delle amicizi degli utenti
        """
        friendships={user:set(friends) for user,friends in self.dataFrame.drop_duplicates(subset="user_id")[["user_id","friends"]].values}
        return friendships

    def printValuesDataset(self):
        dfFriends=self.dataFrame.drop_duplicates(subset="user_id")[["user_id","friends"]]
        numArchiUserFriend=sum([len(listFriends) for listFriends in dfFriends["friends"].values])/2
        print("\n******* DataFrame Merge ********")
        print("Numero di Business considerati: {}".format(self.getNumBusiness()))
        print("Numero di Utenti considerati: {}".format(self.getNumUsers()))
        print("Numero di Ratings considerati: {}".format(self.getDataFrame().shape[0]))
        print("SPARSITA: {}".format(1-(self.getDataFrame().shape[0]/(self.getNumBusiness()*self.getNumUsers()))))
        print("Numero di Tags considerati: {}".format(self.getNumCatBus()))
        print("Numero di Relazioni (User-User) presenti: {}".format(numArchiUserFriend))

    def getNumUsersWithFriends(self,numFriends):
        # Ritorna il numero di utenti che hanno almeno numFriends
        return sum([1 for lista in self.dataFrame[["user_id","friends"]].drop_duplicates(subset="user_id")["friends"].values if len(lista)>=numFriends])

    def getDistrFriends(self,numFriends):
        # Ritorna la distribuzione di utenti con almeno numFriends
        distrFriends=Counter([len(lista) for lista in self.dataFrame[["user_id","friends"]].drop_duplicates(subset="user_id")["friends"].values if len(lista)>=numFriends])
        return sorted(distrFriends.items(), key=operator.itemgetter(0),reverse=True)

    def setProperties(self,dfMerge):
        self.setNumBusiness(dfMerge["business_id"].unique().shape[0])
        self.setNumUsers(dfMerge["user_id"].unique().shape[0])
        self.setNumCatBus(len({category for listCategories in dfMerge["categories"].values for category in listCategories}))
        self.setDataFrame(dfMerge)

    def setDataFrame(self, dfMergeRB):
        self.dataFrame=dfMergeRB

    def setNumBusiness(self,numBusiness):
        self.numBusiness=numBusiness

    def setNumUsers(self,numUsers):
        self.numUsers=numUsers

    def setNumCatBus(self, numCatBus):
        self.numCatBus=numCatBus

    def getDataFrame(self):
        return self.dataFrame

    def getNumBusiness(self):
        return self.numBusiness

    def getNumUsers(self):
        return self.numUsers

    def getNumCatBus(self):
        return self.numCatBus











