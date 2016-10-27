__author__ = 'maury'

from statistics import mean

from recommender import Recommender

class Printer:
    def __init__(self):
        pass

    def printRecVal(self,rs):
        """
        Stampo dei risultati finali derivanti dalla valutazione del Recommender
        :param rs: Recommender preso in esame
        :type rs: Recommender
        :return:
        """
        print("\n\n************ Risultati Finali *************")
        print("Numero medio ratings da predire: {}".format(mean(rs.getDataEval()["nTestRates"])))
        print("Numero medio ratings personalizzati: {}".format(mean(rs.getDataEval()["nPredPers"])))
        print("MAE: {}".format(mean(rs.getDataEval()["mae"])))
        print("RMSE: {}".format(mean(rs.getDataEval()["rmse"])))
        print("Precision: {}".format(mean(rs.getDataEval()["precision"])))
        print("Recall: {}".format(mean(rs.getDataEval()["recall"])))
        print("F1: {}".format(mean(rs.getDataEval()["f1"])))


if __name__ == '__main__':
    pass
