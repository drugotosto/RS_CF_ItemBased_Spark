__author__ = 'maury'

from sklearn.metrics import mean_absolute_error,mean_squared_error
from statistics import mean

from conf.confRS import topN

class Evaluator:
    def __init__(self):
        # Dizionario che rappresenta i dati che compongono il TestSet (verrà settato più avanti)
        self.test_ratings=None
        # Risultati derivanti dalla valutazione medie delle diverse metriche utilizzate sui folds
        self.dataEval={"nTestRates":[],"nPredPers":[],"mae":[],"rmse":[],"precision":[],"recall":[],"f1":[],"covUsers":[],"covMedioBus":[]}


    def computeEvaluation(self,dictRec,analyzer):
        """
        Calcolo delle diverse misure di valutazione per il dato Recommender passato in input per un certo fold
        :param dictRec: Dizionario che per ogni user contiene una lista di tutte le raccomandazioni possibili su items ordinati -> user:[(scorePred,item),(scorePred,item),...]
        :param analyzer: Analizzatore del DataSet originale dato in input
        :type analyzer: DataScienceAnalyzer
        """
        precisions=[]
        recalls=[]
        listMAEfold=[]
        listRMSEfold=[]
        # Numero di predizioni personalizzate che si è stati in grado di fare su tutto il fold
        nPredPers=0
        nUserPers=0
        # Ciclo su ogni user del test per recuperare la lista di Predizioni da testare: lista di coppie [(ratePred,rateTest)...]
        for userTest,ratingsTest in self.test_ratings.items():
            # Controllo se per il suddetto utente è possibile effettuare una predizione personalizzata (la lista di suggerimenti contiene almeno 1 elemento)
            if userTest in dictRec and len(dictRec[userTest])>0:
                nUserPers+=1
                # Lista di coppie di (TrueRates,PredRates) prese in esame per l'active user
                pairsRatesPers=[]
                # Numero totale di items ritenuti rilevanti dall'utente
                nTotRil=0
                # Numero di items tra quelli ritenuti rilevanti dall'utente che fanno anche parte della lista dei suggerimenti
                numTorRil=0
                predRates,items=zip(*dictRec[userTest])
                # Ciclo su tutti gli items per cui devo predire il rate
                for item,rate in ratingsTest:
                    # Controllo che l'item sia tra quelli per cui si è fatta una predizione (fa parte di uno di quelli della lista dei suggerimenti)
                    if item in items:
                        # Aggiungo la coppia (ScorePredetto,ScoreReale) utilizzata per MAE,RMSE
                        pairsRatesPers.append((predRates[items.index(item)],rate))
                        # Tenendo conto di tutta la lista dei suggerimenti vedo se l'item votato realmente dall'active user risulta presente
                        nPredPers+=1

                    # Controllo se l'item risulta essere rilevante
                    if rate>=3:
                        nTotRil+=1
                        #  Controllo nel caso sia presente nei TopN suggeriti
                        if item in items[:topN]:
                            numTorRil+=1

                if pairsRatesPers:
                    # Calcolo MAE,RMSE (personalizzato) per un certo utente per tutti i suoi testRates
                    trueRates=[elem[0] for elem in pairsRatesPers]
                    predRates=[elem[1] for elem in pairsRatesPers]
                    mae=mean_absolute_error(trueRates,predRates)
                    listMAEfold.append(mae)
                    rmse=mean_squared_error(trueRates,predRates)
                    listRMSEfold.append(rmse)

                # Controllo se tra i rates dell'utente usati come testSet ci sono anche rates di items ritenuti Rilevanti
                if nTotRil>0:
                    # Calcolo della RECALL per il tale utente sotto esame
                    recalls.append(numTorRil/nTotRil)
                    # Calcolo della PRECISION per il tale utente sotto esame
                    precisions.append(numTorRil/topN)

        """************** Calcolo delle CoverageItems/CoverageUsers *****************"""
        percUsers,percMedioBus=self.computeCoverage(analyzer,dictRec,nUserPers)

        """
        dictBusCat=analyzer.getDictBusCat()
        print("\nSpazio MEMORIA: {}".format(deep_getsizeof(dictBusCat,set())))
        """
        # Registro le valutazioni appena calcolare per il fold preso in considerazione
        self.appendMisuresFold(nPredPers,listMAEfold,listRMSEfold,recalls,precisions,percUsers,percMedioBus)


    def computeCoverage(self,analyzer,dictRec,nUserPers):
        """
        Computazione dello User Space Covarage e dello Item Catalog Coverage
        :param analyzer: Analizzatore del DataSet originale dato in input
        :param dictRec: Dizionario che per ogni user contiene una lista di tutte le raccomandazioni possibili su items ordinati -> user:[(scorePred,item),(scorePred,item),...]
        :return: Percentuale degli utenti del fold per cui è stata prodotto almeno un suggerimento, Copertura media in percentuale (sugli users del fold) del catalogo degli items
        """
        dictUserPercBus={user:len(set([pair[1] for pair in listaPair]))/analyzer.getNumBusiness() for user,listaPair in dictRec.items() if len(listaPair)>0}
        """ User Space Coverage """
        # Percetuale di utenti (per il fold considerato) per i quali è stato possibile riporate una lista dei suggerimenti non vuota
        percUsers=nUserPers/self.nTestUsers
        # Media delle percentuali (per il fold considerato) di copertura dei business su tutti gli utenti del fold
        """ Catalogo Coverage """
        percMedioBus=mean(dictUserPercBus.values())
        return percUsers,percMedioBus

    def appendMisuresFold(self,nPredPers,listMAEfold,listRMSEfold,recalls,precisions,percUsers,percMedioBus):
        """
        Aggiungo per ogni fold i valori delle diverse metriche calcolate
        :param nPredPers: Numero delle predizioni personalizzate che sono stato in grado di eseguire
        :param listMAEfold: Media sugli utenti del fold del MAE
        :param listRMSEfold: Media sugli utenti del fold del RSME
        :param recalls: Media sugli utenti del Fold del valore di RECALL
        :param precisions: Media sugli utenti del Fold del valore di PRECISION
        :param percUsers: Percentuale di utenti sul fold che hanno avuto una lista dei suggerimenti non vuota
        :param percMedioBus: Copertura media in percentuale (considerando tutti fli utenti del fold) del catalogo degli items
        """
        self.dataEval["nPredPers"].append(nPredPers)
        # Calcolo del valore medio di MAE,RMSE sui vari utenti appartenenti al fold
        # print("MAE (personalizzato) medio fold: {}".format(mean(listMAEfold)))
        self.dataEval["mae"].append(mean(listMAEfold))
        # print("RMSE (personalizzato) medio fold: {}".format(mean(listRMSEfold)))
        self.dataEval["rmse"].append(mean(listRMSEfold))
        # Calcolo del valore medio di precision e recall sui vari utenti appartenenti al fold
        # print("MEAN RECALL: {}".format(mean(recalls)))
        self.dataEval["recall"].append(mean(recalls))
        # print("MEAN PRECISION: {}".format(mean(precisions)))
        self.dataEval["precision"].append(mean(precisions))
        f1=(2*mean(recalls)*mean(precisions))/(mean(recalls)+mean(precisions))
        # print("F1 FOLD: {}".format(f1))
        self.dataEval["f1"].append(f1)
        # print("\nAl {} % di Users riusciamo a fornire dei suggerimenti per 'mediamente' il {} % dei Business totali".format(percUsers,percMedioBus))
        self.dataEval["covMedioBus"].append(percMedioBus)
        self.dataEval["covUsers"].append(percUsers)

    def setTestRatings(self,test_ratings):
        self.test_ratings=test_ratings

    def appendNtestRates(self,nTestRates):
        self.dataEval["nTestRates"].append(nTestRates)

    def getDataEval(self):
        return self.dataEval

    def setNumTestUsers(self,nTestUsers):
        self.nTestUsers=nTestUsers




