__author__ = 'maury'

from conf.confRS import categoria
from conf.confSocialItemBased import onlyFriends


""" File di configurazione generico delle diverse directories e files utilizzati. """

# Directory (base) che contiene i diversi files Json dai quali poter costruire il Dataframe/RDD di input del RS
directoryDataSets="/home/maury/Desktop/Datasets/Yelp/Dataset/yelp_dataset_challenge_academic_dataset/"
# Directory contenente i  diversi files (train e test) di input per la K-Fold Cross-Validation
dirPathInput="/home/maury/Desktop/SparkSets/"+categoria+"/"
if onlyFriends:
    dirPathInput=dirPathInput+"FriendsOnly/"
else:
    dirPathInput=dirPathInput+"Base/"
# Directory contenente i  diversi files (ognuno per un certo Recommender) che contengono i risultati della K-Fold Cross-Validation
dirPathOutput="/home/maury/Desktop/SparkOutput/"+categoria+"/"
if onlyFriends:
    dirPathOutput=dirPathOutput+"FriendsOnly/"
else:
    dirPathOutput=dirPathOutput+"Base/"
# Directory contenete i diversi Folds di train/test sets
dirFolds=dirPathInput+"Folds/"
# Directory contenente i files che a loro volta contengono i dati di un testSet
dirTest=dirFolds+"test_"
# Directory contenente i files che a loro volta contengono i dati di un testSet
dirTrain=dirFolds+"train_"

# File delle Reviews
reviewsJSON=directoryDataSets+"yelp_academic_dataset_review.json"
# File dei Business
businessJSON=directoryDataSets+"yelp_academic_dataset_business.json"
# File dei Users
usersJSON=directoryDataSets+"yelp_academic_dataset_user.json"
# File del DataSet di partenza da cui si andranno a creare i files di Train/Test
datasetJSON=dirPathInput+"dataSetRatings.json"
