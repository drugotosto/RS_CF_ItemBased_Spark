import os

"""
Esecuzione di un file pyhton "link_clustering.py" per trovare le diverse commmunities di nodi con possibile overlappings

File che dato in input edgelist.json ritona 3 file:
    - edge2comm, an edge on each line followed by the community id (cid) of the edge's link comm:
        node_i <delimiter> node_j <delimiter> cid <newline>

    - comm2edges, a list of edges representing one community per line:
        cid <delimiter> ni,nj <delimiter> nx,ny [...] <newline>

    - comm2nodes, a list of nodes representing one community per line:
        cid <delimiter> ni <delimiter> nj [...] <newline>

  The output filename contains the threshold at which the dendrogram
  was cut, if applicable, or the threshold where the maximum
  partition density was found, and the value of the partition
  density.

  If no threshold was given to cut the dendrogram, a file ending with
  `_thr_D.txt' is generated, containing the partition density as a
  function of clustering threshold.

    P.S: occorre utilizzare interprete python2
"""

if __name__ == '__main__':
    os.system("python2 /home/maury/Desktop/ClusteringMethods/LinkClustering/link_clustering.py /home/maury/Desktop/ClusteringMethods/LinkClustering/edgelist.json")