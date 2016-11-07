__author__ = 'maury'

from pandas import Series, DataFrame
from itertools import islice
import pandas as pd
import json

from conf.confDirFiles import reviewsJSON, businessJSON

class DataScienceAnalyzer():
    def __init__(self,categoria):
        self.dataFrame=None
        self.dictBusCat=None
        self.categoria=categoria
        self.numUsers=None
        self.numBusiness=None
        self.numCatBus=None

    def createDataSet(self):
        # LETTURA DEL FILE JSON DELLE REVIEWS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME CHE SERVIRÀ IN FASE DI PREDIZIONE DEI RATES
        dfRatings=self.createRatingsDF()
        # LETTURA DEL FILE JSON DEI BUSINESS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME
        dfBusiness=self.createBusinessDF()
        dictBusCat=dfBusiness["categories"].to_dict()
        self.setDictBusCat(dictBusCat)
        print("\nDizionario delle categorie dei Business creato!")
        # Merge dei dataframe tra RATINGS e BUSINESS e Selezione dei soli ratings appartenenti ad una specifica categoria
        dfMergeRB=pd.merge(dfRatings,dfBusiness,left_on='business_id',right_index=True, how="inner")
        dfMergeRB=dfMergeRB[dfMergeRB["categories"].apply(lambda x: bool(set(x).intersection([self.categoria])))]
        # Vado a settare i parametri del DataFrame che riguardano: Numero di Business, Numero di Users, Numero di Categorie di Business
        self.setNumBusiness(dfMergeRB["business_id"].unique().shape[0])
        self.setNumUsers(dfMergeRB["user_id"].unique().shape[0])
        self.setNumCatBus(len({category for listCategories in dfMergeRB["categories"].values for category in listCategories}))
        print("\nDataFrame Merge finale creato!")
        self.setDataFrame(dfMergeRB)
        return dfMergeRB

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