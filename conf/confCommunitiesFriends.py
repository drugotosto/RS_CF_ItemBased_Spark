__author__ = 'maury'

from conf.confDirFiles import dirPathCommunities

# Elenco di tutte gli algoritmi di detection community possibili
communitiesTypes=["fastgreedy", "walktrap", "label_propagation", "infomap", "multilevel"]
# Scelta di uno tra i vari algortimi di detection communities ("tipo"/"all")
communityType="multilevel"

# File delle communities scelto da utilizzare dal quale andare a calcolare le somiglianze tra utenti
fileFriendsCommunities=dirPathCommunities+"/"+communityType+"/communitiesFriends.json"