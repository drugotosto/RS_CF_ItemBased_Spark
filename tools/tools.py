__author__ = 'maury'

import os
import json
from statistics import mean
import shutil

from tools.evaluator import Evaluator


def printRecVal(evaluator,directory,fileName):
    """
    Salvataggio su file dei risultati finali derivanti dalla valutazione del Recommender.
    :param evaluator: Evaluator che contiene i risultati derivanti dalla valutazione del Recommender in esame
    :type evaluator: Evaluator
    :param fileName: Nome del file su cui andare a scrivere i risultati ottenuti
    :return:
    """
    # Creazione della directory se ancora assente
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName+".json","w") as f:
        f.write(json.dumps({v:mean(k) for v,k in evaluator.getDataEval().items()}))

def saveJsonData(data,directory,fileName):
    """
    Salva sul file (un elemento per riga) i dati passati in formato json
    :param data: Dati da salvare su file in json
    :param fileName: Nome del file su cui andare a salvare i dati
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fileName,"w") as f:
        for dato in data:
            f.write(json.dumps(dato)+"\n")

if __name__ == '__main__':
    """ Possibilità di chiamare altri metodi per stampare dati/grafici e confrontate le prestazioni dei diversi Recommenders. """
    pass