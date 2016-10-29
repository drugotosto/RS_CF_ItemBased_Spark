__author__ = 'maury'

""" File di configurazione generico. """

# Numero di Folds in cui si dividono gli users/ratings per effettuare la valutazione del RS (K-Fold Cross-Validation)
nFolds=5
# Parametro relativo al numero di elementi della lista dei sugerimenti finali rilasciati all'utente
topN=10
# Directory contenente i  diversi files (train e test) di input per la K-Fold Cross-Validation
dirPathInput = "/home/maury/Desktop/SparkSets/"
# Directory contenente i  diversi files (ognuno per un certo Recommender) che contengono i risultati della K-Fold Cross-Validation
dirPathOutput="/home/maury/Desktop/SparkOutput/"