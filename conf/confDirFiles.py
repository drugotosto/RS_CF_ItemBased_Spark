__author__ = 'maury'

from conf.confRS import tag

""" File di configurazione generico delle diverse directories e files utilizzati. """

# Directory (base) che contiene i diversi files Json dai quali poter costruire il Dataframe/RDD (filtrato) di input del RS
directoryDataSets="/home/maury/Desktop/Datasets/Yelp/Dataset/yelp_dataset_challenge_academic_dataset/"
# Directory di lavoro principale della categoria selezionata che contiene i files tramite i quali ci si basa per fare recommandations
dirPathInput="/home/maury/Desktop/SparkSets/"+tag+"/"
# Directory che contiene tutti le diverse communities di amici trovare a seconda dell'algortimo utilizzato
dirPathCommunities=dirPathInput+"Communities/"
# Directory che contiene un file per ogni tipologia di Recommender di cui si sono calcolati i risultati della K-Fold Cross-Validation
dirPathOutput="/home/maury/Desktop/SparkOutput/"+tag+"/"

# Directory con i diversi Folds di train/test sets per la K-Fold Cross-Validation
dirFolds=dirPathInput+"Folds/"
# Directory che contiene i files di test di una delle prove di Cross-Validation
dirTest=dirFolds+"test_"
# Directory che contiene i files di train di una delle prove di Cross-Validation
dirTrain=dirFolds+"train_"

# File delle Reviews
reviewsJSON=directoryDataSets+"yelp_academic_dataset_review.json"
# File dei Business
businessJSON=directoryDataSets+"yelp_academic_dataset_business.json"
# File dei Users
usersJSON=directoryDataSets+"yelp_academic_dataset_user.json"
# File dei ratings (filtrati) del Dataset di riferimento
datasetJSON=dirPathInput+"dataSetRatings.json"
# File Json che associa ad ogni utente l'insieme dei Tags dei business da lui votati con peso associato
userTagJSON=dirPathInput+"userTag.json"
# File Grafo pesato delle amicizie tra utenti
userFriendsGraph=dirPathInput+"userFriendsGraph"

