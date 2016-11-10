__author__ = 'maury'

""" File di configurazione generico del RS. """
typeRecommender="ItemBased"
# Numero di Folds in cui si dividono gli users/ratings per effettuare la valutazione del RS (K-Fold Cross-Validation)
nFolds=5
# Parametro relativo al numero di elementi della lista dei sugerimenti finali rilasciati all'utente
topN=10
# Scelta della categoria su cui andare a costruire il corrispondente DataFrame/RDD
categoria="Nightlife"
# Percentuale del numero di rates da prendere (per ogni utente) che faranno parte del test set sul numero totale di ratings disponibili
percTestRates=0.1
# Scelta relativa al fatto di andare a considerare solamente i ratings degli amici dell'utente o meno
onlyFriends=False

