__author__ = 'maury'

import json

from conf.confRS import topN,dirPath
from sparkEnvLocal import SparkEnvLocal

class Recommender:

    def __int__(self,spEnv):
        """
        :param spEnv: Oggetto che rappresenta lo SparkEnviroment
        :type spEnv: SparkEnvLoc
        :return:
        """
        self.topN=topN
        self.dirPath=dirPath
        # Verrà settato più avanti
        self.user_item_pairs=None
        if spEnv:
            self.sc=spEnv.getSc()

    def retiveData(self,fold):
        """
        Recupero i dati (dai files) su cui poi andare a costruire il modello
        :return:
        """
        fileName = self.dirPath+"trainSetFold_"+str(fold)+".json"
        # Recupero i dati del TrainSet creando la conseguente "matrice" dei Rate raggruppando i rates dei vari utenti
        lines = self.sc.textFile(fileName)
        # Costruisco un pairRDD del tipo (user,[(item,rate),(item,rate),...]) e lo rendo persistente
        user_item_pairs = lines.map(self._parseFile).groupByKey().cache()
        self.setUser_item_pairs(user_item_pairs)

    def _parseFile(line):
        """
        Parsifico ogni linea del file e costruisco (user,(item,rate))
        """
        jsonObj = json.loads(line)
        return jsonObj[0],(jsonObj[1],float(jsonObj[2]))

    def builtModel(self):
        """
        Costruzione del modello a seconda dell'approccio utilizzato
        :return:
        """
        pass

    def saveData(self,ris):
        """
        Salvataggio dei dati derivanti dalla valutazione fatta
        :param ris:
        :return:
        """
        pass

    def setUser_item_pairs(self, user_item_pairs):
        self.user_item_pairs=user_item_pairs


