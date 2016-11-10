from tools.tools import deep_getsizeof

__author__ = 'maury'

from sklearn.metrics import mean_absolute_error,mean_squared_error
from statistics import mean

from conf.confRS import topN

class Evaluator:
    def __init__(self):
        # Dizionario che rappresenta i dati che compongono il TestSet (verrà settato più avanti)
        self.test_ratings=None
        # Risultati derivanti dalla valutazione medie delle diverse metriche utilizzate sui folds
        self.dataEval={"nTestRates":[],"nPredPers":[],"mae":[],"rmse":[],"precision":[],"recall":[],"f1":[],"covUsers":[],"covMedioBus":[],"diversity":[]}


    def setTestRatings(self,test_ratings):
        self.test_ratings=test_ratings

    def appendNtestRates(self,nTestRates):
        self.dataEval["nTestRates"].append(nTestRates)

    def computeEvaluation(self,sc,dictRec,analyzer):
        """
        Calcolo delle diverse misure di valutazione per il dato Recommender passato in input per un certo fold
        :param sc: SparkContext
        :type sc: SparkContext
        :param dictRec: Dizionario che per ogni user contiene una lista di predizioni su items ordinati [(scorePred,item),(scorePred,item),...]
        :param analyzer: Analizzatore del DataSet originale dato in input
        :type analyzer: DataScienceAnalyzer
        :return:
        """
        precisions=[]
        recalls=[]
        listMAEfold=[]
        listRMSEfold=[]
        listDiversityFold=[]
        # Numero di predizioni personalizzate che si è stati in grado di fare su tutto il fold
        nPredPers=0
        # Ciclo sul dizionario del test per recuperare le coppie (ratePred,rateTest)
        for userTest,ratingsTest in self.test_ratings.items():
            # Controllo se per il suddetto utente è possibile effettuare una predizione personalizzata
            if userTest in dictRec and len(dictRec[userTest])>0:
                # Coppie di (TrueRates,PredRates) preso in esame il tale utente
                pairsRatesPers=[]
                # Numero di items tra quelli ritenuti rilevanti dall'utente che sono stati anche fatti tornare
                numTorRil=0
                # Numero totale di items ritenuti rilevanti dall'utente
                nTotRil=0
                predRates,items=zip(*dictRec[userTest])
                # Ciclo su tutti gli items per cui devo predire il rate
                for item,rate in ratingsTest:
                    # Controllo che l'item sia tra quelli per cui si è fatta una predizione
                    if item in items:
                        # Aggiungo la coppia (ScorePredetto,ScoreReale) utilizzata per MAE,RMSE
                        pairsRatesPers.append((predRates[items.index(item)],rate))
                        nPredPers+=1

                    # Controllo se l'item risulta essere rilevante
                    if rate>3:
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
        percUsers,percMedioBus=self.computeCoverage(analyzer,dictRec)

        """*************************** Calcolo Diversity ***********************"""
        # business=sc.parallelize(dictRec.items()).filter(lambda x: len(x[1]) > 1).map(lambda x: Evaluator.collectBus(x[1]))
        # dictBusCat=sc.broadcast(analyzer.getDictBusCat())
        # numCatBus=analyzer.getNumCatBus()
        # listDiversity=business.map(lambda x: Evaluator.computeDiversity(x,dictBusCat.value,numCatBus)).collect()
        # diversity=sum(listDiversity)/len(listDiversity)
        #
        """
        dictBusCat=analyzer.getDictBusCat()
        print("\nSpazio MEMORIA: {}".format(deep_getsizeof(dictBusCat,set())))
        """
        diversity=1
        # Registro le valutazioni appena calcolare per il fold preso in considerazione
        self.appendMisuresFold(nPredPers,listMAEfold,listRMSEfold,recalls,precisions,percUsers,percMedioBus,diversity)


    def computeCoverage(self,analyzer,dictRec):
        # ************************ CoverageItems ***********************
        # _,items=zip(*[pair for user,listaPair in dictRec.items() for pair in listaPair if listaPair])
        # numBusinessPers=len(set(items))
        # percBus=numBusinessPers/analyzer.getNumBusiness()

        # *********************** Coverage Users/Items ************************
        dictUserPercBus={user:len(set([pair[1] for pair in listaPair]))/analyzer.getNumBusiness() for user,listaPair in dictRec.items() if listaPair}
        percUsers=len(dictUserPercBus)/analyzer.getNumUsers()
        percMedioBus=sum(dictUserPercBus.values())/len(dictUserPercBus)
        return percUsers,percMedioBus

    @staticmethod
    def computeDiversity(listBusiness, dictBusCat, numCatBus):
        """
        Calcolo della dìversità media degi Business presenti nella Top-N di ogni utente
        :param dictRec: Dizionario che per ogni user contiene una lista di predizioni su items ordinati (user:[(scorePred,item),(scorePred,item),...])
        :param topN: Parametro che definisce lunghezza della lista dei Business suggeriti
        :return: Valore medio di Diversity
        """
        return len({cat for bus in listBusiness for cat in dictBusCat[bus]})/numCatBus

    def appendMisuresFold(self,nPredPers,listMAEfold,listRMSEfold,recalls,precisions,percUsers,percMedioBus,diversity):
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
        # Calcolo del valore medio di diversity per il dato Fold mediato su tutti gli users
        self.dataEval["diversity"].append(diversity)

    def getDataEval(self):
        return self.dataEval

    @staticmethod
    def collectBus(listPairs):
        business=[]
        if listPairs:
            _,business=zip(*listPairs)
        return business[:topN]


