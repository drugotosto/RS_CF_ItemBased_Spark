__author__ = 'maury'

from pandas import Series, DataFrame
from itertools import islice
import pandas as pd
import json

from conf.confRS import reviewsJSON, businessJSON

class DataScienceAnalyzer():
    def __init__(self,categoria):
        self.categoria=categoria


    def createDataSet(self):
        # LETTURA DEL FILE JSON DELLE REVIEWS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME CHE SERVIRÀ IN FASE DI PREDIZIONE DEI RATES
        dfRatings=self.createRatingsDF()
        # LETTURA DEL FILE JSON DEI BUSINESS A PEZZI PER LA COSTRUZIONE DEL DATAFRAME
        dfBusiness=self.createBusinessDF()
        # Merge dei dataframe tra RATINGS e BUSINESS e Selezione dei soli ratings appartenenti ad una specifica categoria
        dfMergeRB=pd.merge(dfRatings,dfBusiness,left_on='business_id',right_index=True, how="inner")
        dfMergeRB=dfMergeRB[dfMergeRB["categories"].apply(lambda x: bool(set(x).intersection([self.categoria])))]
        print("\nDataFrame Merge finale creato!")
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

if __name__ == '__main__':
    pass