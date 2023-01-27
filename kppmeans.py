import numpy as np
import random
import sys

data = []

centroids = []
clusters = []


def distance(point, centroid):
    euc_dist = 0
    for x in range(len(point)): 
        euc_dist+=(point[x]-centroid[x])**2
    return euc_dist


distances = []
def min_dist_to_centroid():
    for point in data:
        temp_dist = []
        for centroid in centroids:
            temp_dist.append(distance(point,centroid))
        distances.append(min(temp_dist))

def normalize_distances():
    vector_norm = np.linalg.norm(distances, ord=1)
    normalized_distances = distances/vector_norm
    return normalized_distances

def pick_weighted_rnd(normalized_distances):
    draw = np.random.choice(np.arange(0,len(data)), p=normalized_distances)
    centroids.append(data[draw])


def calculate_centroid(index):
    mean = []
    for element in clusters[index]:
        if(len(mean)==0):
            mean = clusters[index][0].copy()
        else:
            for i in range(len(element)):
                mean[i]+=element[i]
                
    for i in range(len(mean)):
        mean[i]= mean[i]/len(clusters[index])

    centroids[index] = mean


def kplusplusmeans(k, random_int):
    global distances
    first_centroid = data[random_int]
    centroids.append(first_centroid)
    for x in range(k):
        distances = []
        min_dist_to_centroid()
        normalized_distances = normalize_distances()
        pick_weighted_rnd(normalized_distances)

def converged(prev_centroids):
    total_movement = 0
    for i in range(len(centroids)):
        total_movement += distance(centroids[i], prev_centroids[i])
    if(total_movement == 0):
        return True
    else:
        return False

def cluster(max_iterations):
    global clusters
    global centroids
    for iteration in range(max_iterations):
        clusters = [[] for x in range (len(centroids)) ]
        for element in data:
            dist_to_centroids = []
            for centroid in centroids:
                dist_to_centroids.append(distance(element,centroid))
            index_of_centroid = dist_to_centroids.index(min(dist_to_centroids))
            clusters[index_of_centroid].append(element)
        prev_centroids = centroids.copy()    
        for i in range(len(centroids)):
            calculate_centroid(i)
        if(converged(prev_centroids)):
            print("iterations:",iteration+1)
            break



def main():
    global data
    path = sys.argv[1]
    k = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    data = np.genfromtxt(path,delimiter='\t',dtype=float)
    #
    random_int = random.randint(0, len(data)-1)
    kplusplusmeans(k-1,random_int)
    cluster(max_iterations)
    
    for i in range(len(clusters)):
        print("cluster",i+1)
        print("size:",len(clusters[i]))
        for element in clusters[i]:
            print(element)
        print("\n")
        
if __name__ == "__main__":
    main()
