__author__ = 'maury'

import os
import json
from collections import Mapping, Container
from sys import getsizeof
from statistics import mean


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
    print("DATA EVAL: {}".format(evaluator.getDataEval()))
    with open(fileName+".json","w") as f:
        # Calcolo della media dei valori trovati su tutti i fold
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

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

if __name__ == '__main__':
    """ Possibilità di chiamare altri metodi per stampare dati/grafici e confrontate le prestazioni dei diversi Recommenders. """
    pass