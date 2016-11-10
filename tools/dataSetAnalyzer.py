__author__ = 'maury'

from pandas import Series, DataFrame
from itertools import islice
import pandas as pd
import json

from conf.confDirFiles import reviewsJSON, businessJSON, usersJSON


class DataScienceAnalyzer():
    def __init__(self,categoria):
        self.dataFrame=None
        self.dictBusCat=None
        self.categoria=categoria
        self.numUsers=None
        self.numBusiness=None
        self.numCatBus=None

    def createDataSet(self,onlyFriends):
        # LETTURA DEL FILE JSON DELLE REVIEWS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME CHE SERVIRÀ IN FASE DI PREDIZIONE DEI RATES
        dfRatings=self.createRatingsDF()
        # LETTURA DEL FILE JSON DEI BUSINESS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME
        dfBusiness=self.createBusinessDF()
        dictBusCat=dfBusiness["categories"].to_dict()
        self.setDictBusCat(dictBusCat)
        print("\nDizionario delle categorie dei Business creato!")
        # Merge dei dataframe tra RATINGS e BUSINESS e Selezione dei soli ratings appartenenti ad una specifica categoria
        dfMerge=pd.merge(dfRatings,dfBusiness,left_on='business_id',right_index=True, how="inner")
        if onlyFriends:
            dfUsers=self.createUsersDF()
            dfMerge=pd.merge(dfMerge,dfUsers,left_on='user_id',right_index=True, how="inner")
        dfMerge=dfMerge[dfMerge["categories"].apply(lambda x: bool(set(x).intersection([self.categoria])))]
        # Vado a settare i parametri del DataFrame che riguardano: Numero di Business, Numero di Users, Numero di Categorie di Business
        self.setNumBusiness(dfMerge["business_id"].unique().shape[0])
        self.setNumUsers(dfMerge["user_id"].unique().shape[0])
        self.setNumCatBus(len({category for listCategories in dfMerge["categories"].values for category in listCategories}))
        print("\nDataFrame Merge finale creato!")
        self.setDataFrame(dfMerge)
        return dfMerge

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

    def setDataFrame(self, dfMergeRB):
        self.dataFrame=dfMergeRB

    def setDictBusCat(self,dictBusCat):
        self.dictBusCat=dictBusCat

    def getDictBusCat(self):
        return self.dictBusCat

    def getCatBusiness(self, listBusiness):
        """
        Recupero tutti le categorie differenti associate ai vari business
        :param businessPred: Lista di business suggeriti all'utente
        :type businessPred: list
        :return: Numero di categorie differenti
        """
        listCategories=[self.dataFrame[self.dataFrame["business_id"]==business].head(1).categories.values[0] for business in listBusiness]
        numCategories=len(set().union(*listCategories))
        return numCategories

    def getFriends(self):
        friends={user:set(friends) for user,friends in self.dataFrame.drop_duplicates(subset="user_id")[["user_id","friends"]].values}
        return friends

    def setNumBusiness(self,numBusiness):
        self.numBusiness=numBusiness

    def setNumUsers(self,numUsers):
        self.numUsers=numUsers

    def setNumCatBus(self, numCatBus):
        self.numCatBus=numCatBus

    def getNumBusiness(self):
        return self.numBusiness

    def getNumUsers(self):
        return self.numUsers

    def getNumCatBus(self):
        return self.numCatBus

if __name__ == '__main__':
    pass