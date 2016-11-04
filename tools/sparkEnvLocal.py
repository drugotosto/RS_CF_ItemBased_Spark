__author__ = 'maury'

from pyspark import SparkContext, SparkConf

class SparkEnvLocal:
    def __init__(self):
        """
        Vado a settare lo SparkContext
        :return:
        """
        # Inizializzazione dello SparkContext
        conf = SparkConf().setMaster("local[8]").setAppName("RecommenderSystem")
        self.sc = SparkContext(conf=conf)

    def getSc(self):
        return self.sc