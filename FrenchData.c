#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
// #include <svm.h>
// #include <gsl/gsl_matrix.h>
// #include <gsl/gsl_eigen.h>


#define MAX_LINE_LENGTH 1024
#define MAX_FEATURES 10
#define K_NEIGHBORS 5 // Number of neighbors for LOF calculation
#define LOF_THRESHOLD 1.5 // Example threshold

typedef struct {
    float features[MAX_FEATURES]; // Store features as an array of floats
} WeatherData;

typedef struct {
    float distance;
} Neighbor;


typedef struct Node {
    int featureIndex;     // Index of the feature to split on
    float splitValue;     // Value to split on at the featureIndex
    struct Node* left;    // Pointer to left subtree
    struct Node* right;   // Pointer to right subtree
} Node;


size_t numFeatures = MAX_FEATURES; 
float euclideanDistance(float* point1, float* point2, size_t num_features);
int knnClassify(WeatherData* dataset, size_t dataSize, float* queryPoint, int k, size_t num_features);
float calculateABOD(float* point, WeatherData* dataset, size_t dataSize, size_t num_features);
float calculateDistance(float* point1, float* point2, size_t num_features);
float calculateAvgDistance(WeatherData* dataset, size_t dataSize, float* point, int k, size_t num_features);
float calculateCBLOF(WeatherData* dataset, size_t dataSize, float* point, int k, size_t num_features);
float randomFloat(float min, float max);
int* generateRandomFeatures(size_t num_features, size_t subsample_size);
Node* buildIsolationTree(WeatherData* dataset, size_t dataSize, size_t max_height, size_t current_height, int* randomFeatures, size_t subsample_size);
Node** buildIsolationForest(WeatherData* dataset, size_t dataSize, size_t num_trees, size_t subsample_size, size_t max_height);
float calculateAnomalyScore(Node** forest, size_t num_trees, WeatherData* point, size_t num_features, size_t dataSize);
size_t calculatePathLength(Node* node, WeatherData* point, size_t num_features, size_t current_height);
float kthNearestNeighborDistance(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k);
float reachabilityDistance(WeatherData* dataset, size_t dataSize, WeatherData* point1, WeatherData* point2, size_t num_features, int k);
float localReachabilityDensity(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k);
float localOutlierFactor(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k);
// struct svm_node* dataPointToSvmNode(WeatherData* dataPoint);
// svm_model* trainOCSVM(WeatherData* dataset, size_t dataSize);
// void predictOutliers(svm_model* model, WeatherData* dataset, size_t dataSize);
// void computePCA(WeatherData* dataset, size_t dataSize, size_t numFeatures);
void freeIsolationTree(Node* node);

int compareNeighbors(const void* a, const void* b) {
    const Neighbor* na = (const Neighbor*)a;
    const Neighbor* nb = (const Neighbor*)b;

    if (na->distance < nb->distance) return -1;
    else if (na->distance > nb->distance) return 1;
    else return 0;
}


int compareFloats(const void* a, const void* b) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    return (*fa > *fb) - (*fa < *fb);
}



// Dynamically resize the dataset array
WeatherData* resizeDataset(WeatherData* dataset, size_t newSize) {
    WeatherData* newDataset = (WeatherData*)realloc(dataset, newSize * sizeof(WeatherData));
    if (newDataset == NULL) {
        free(dataset);
        fprintf(stderr, "Failed to allocate memory for dataset.\n");
        exit(EXIT_FAILURE);
    }
    return newDataset;
}

// Function to read CSV data, skipping every other line
WeatherData* readCSV(const char* filename, size_t* dataSize) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }

    size_t capacity = 1000; // Initial capacity
    *dataSize = 0; // Number of data points read
    WeatherData* dataset = (WeatherData*)malloc(capacity * sizeof(WeatherData));
    if (!dataset) {
        fclose(file);
        fprintf(stderr, "Failed to allocate memory for dataset.\n");
        exit(EXIT_FAILURE);
    }

    char buffer[MAX_LINE_LENGTH];
    int skipLine = 0;
    while (fgets(buffer, MAX_LINE_LENGTH, file)) {
        if (skipLine) { // Skip every other line
            skipLine = 0;
            continue;
        } else {
            skipLine = 1;
        }
        
        if (*dataSize >= capacity) {
            capacity *= 2; // Double the capacity
            dataset = resizeDataset(dataset, capacity);
        }

        // Assuming the data is separated by commas and is all floats
        char* token = strtok(buffer, ",");
        int featureIndex = 0;
        while (token != NULL && featureIndex < MAX_FEATURES) {
            dataset[*dataSize].features[featureIndex++] = atof(token);
            token = strtok(NULL, ",");
        }
        (*dataSize)++;
    }

    fclose(file);
    return dataset; // Remember to free this memory in the calling function
}


// Assumes the last element in features is the class label for simplicity
int knnClassify(WeatherData* dataset, size_t dataSize, float* queryPoint, int k, size_t num_features) {
    // Temporary structure to store distances and corresponding class labels
    typedef struct {
        float distance;
        float class;
    } Neighbor;

    Neighbor* neighbors = malloc(dataSize * sizeof(Neighbor));
    if (!neighbors) {
        fprintf(stderr, "Failed to allocate memory for neighbors.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate distance from query point to all other points
    for (size_t i = 0; i < dataSize; ++i) {
        neighbors[i].distance = euclideanDistance(dataset[i].features, queryPoint, num_features - 1); // Exclude class label from distance calculation
        neighbors[i].class = dataset[i].features[num_features - 1]; // Assuming last feature is class label
    }

    // Sort neighbors by distance
    qsort(neighbors, dataSize, sizeof(Neighbor), compareNeighbors);




    // Count class occurrences among the k nearest neighbors
    // Assuming classes are integers for simplicity
    int maxClass = (int)neighbors[0].class;
    for (size_t i = 0; i < dataSize; ++i) {
        if ((int)neighbors[i].class > maxClass) {
            maxClass = (int)neighbors[i].class;
        }
    }

    int* classCounts = calloc(maxClass + 1, sizeof(int));
    for (int i = 0; i < k; ++i) {
        classCounts[(int)neighbors[i].class]++;
    }

    // Find the class with the most occurrences
    int predictedClass = 0;
    int maxCount = 0;
    for (int i = 0; i <= maxClass; ++i) {
        if (classCounts[i] > maxCount) {
            predictedClass = i;
            maxCount = classCounts[i];
        }
    }

    free(neighbors);
    free(classCounts);

    return predictedClass;
}

// Euclidean distance function
float euclideanDistance(float* point1, float* point2, size_t num_features) {
    float distance = 0.0;
    for (size_t i = 0; i < num_features; ++i) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

// Function to calculate variance of angles
float calculateABOD(float* point, WeatherData* dataset, size_t dataSize, size_t num_features) {
    float variance = 0.0;
    for (size_t i = 0; i < dataSize; ++i) {
        for (size_t j = i + 1; j < dataSize; ++j) {
            float distance_ij = euclideanDistance(dataset[i].features, dataset[j].features, num_features);
            float distance_ip = euclideanDistance(point, dataset[i].features, num_features);
            float distance_jp = euclideanDistance(point, dataset[j].features, num_features);
            // Calculate cosine of the angle between vectors
            float cosine_angle = ((distance_ip * distance_ip) + (distance_jp * distance_jp) - (distance_ij * distance_ij)) / (2 * distance_ip * distance_jp);
            // Variance calculation
            variance += (1 - cosine_angle) * (1 - cosine_angle);
        }
    }
    return variance;
}

// Function to calculate the Euclidean distance between two points
float calculateDistance(float* point1, float* point2, size_t num_features) {
    float distance = 0.0;
    for (size_t i = 0; i < num_features; ++i) {
        distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(distance);
}

// Function to calculate the average distance from a point to its k nearest neighbors
float calculateAvgDistance(WeatherData* dataset, size_t dataSize, float* point, int k, size_t num_features) {
    // Calculate distances from the point to all other points
    float* distances = malloc(dataSize * sizeof(float));
    if (!distances) {
        fprintf(stderr, "Failed to allocate memory for distances.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < dataSize; ++i) {
        distances[i] = euclideanDistance(dataset[i].features, point, num_features);
    }

    // Sort distances
    qsort(distances, dataSize, sizeof(float), compareFloats); // Assuming compareFloats is correctly defined elsewhere

    // Declare and initialize avgDistance
    float avgDistance = 0.0;
    
    // Calculate the average distance to the k nearest neighbors
    for (int i = 0; i < k; ++i) {
        avgDistance += distances[i];
    }
    avgDistance /= k;

    free(distances); // Don't forget to free allocated memory
    return avgDistance; // Return the calculated average distance
}




// Function to calculate the CBLOF score for a point
float calculateCBLOF(WeatherData* dataset, size_t dataSize, float* point, int k, size_t num_features) {
    // Calculate the average distance to the k nearest neighbors
    float avgDistance = calculateAvgDistance(dataset, dataSize, point, k, num_features);

    // Calculate the reachability of the point
    float reachability = 1.0 / avgDistance;

    // Calculate the sum of reachability for all points
    float sumReachability = 0.0;
    for (size_t i = 0; i < dataSize; ++i) {
        float distance = euclideanDistance(dataset[i].features, point, num_features);
        sumReachability += reachability / distance;
    }

    // Calculate CBLOF score
    float cblof = sumReachability / dataSize;
    return cblof;
}


// Random number generator
float randomFloat(float min, float max) {
    return min + (rand() / (RAND_MAX / (max - min)));
}

// Generate a random feature subset
int* generateRandomFeatures(size_t num_features, size_t subsample_size) {
    int* randomFeatures = malloc(subsample_size * sizeof(int));
    if (!randomFeatures) {
        fprintf(stderr, "Failed to allocate memory for random features.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < subsample_size; ++i) {
        randomFeatures[i] = rand() % num_features;
    }

    return randomFeatures;
}

// Create a new node
Node* createNode(int featureIndex, float splitValue) {
    Node* node = malloc(sizeof(Node));
    if (!node) {
        fprintf(stderr, "Failed to allocate memory for node.\n");
        exit(EXIT_FAILURE);
    }
    node->featureIndex = featureIndex;
    node->splitValue = splitValue;
    node->left = NULL; // Initialize left child pointer to NULL
    node->right = NULL; // Initialize right child pointer to NULL
    return node;
}


// Recursively build an isolation tree
Node* buildIsolationTree(WeatherData* dataset, size_t dataSize, size_t max_height, size_t current_height, int* randomFeatures, size_t subsample_size) {
    if (current_height >= max_height || dataSize <= 1) {
        return NULL;
    }

    // Choose a random feature
    int randomFeatureIndex = randomFeatures[rand() % subsample_size];

    // Randomly choose a split value within the range of the selected feature
    float min_value = INFINITY;
    float max_value = -INFINITY;
    for (size_t i = 0; i < dataSize; ++i) {
        if (dataset[i].features[randomFeatureIndex] < min_value) {
            min_value = dataset[i].features[randomFeatureIndex];
        }
        if (dataset[i].features[randomFeatureIndex] > max_value) {
            max_value = dataset[i].features[randomFeatureIndex];
        }
    }
    float splitValue = randomFloat(min_value, max_value);

    // Partition the dataset based on the split value
    size_t leftSize = 0;
    size_t rightSize = 0;
    WeatherData* leftData = malloc(dataSize * sizeof(WeatherData));
    WeatherData* rightData = malloc(dataSize * sizeof(WeatherData));
    if (!leftData || !rightData) {
        fprintf(stderr, "Failed to allocate memory for left or right data.\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < dataSize; ++i) {
        if (dataset[i].features[randomFeatureIndex] <= splitValue) {
            leftData[leftSize++] = dataset[i];
        } else {
            rightData[rightSize++] = dataset[i];
        }
    }

    // Recursively build left and right subtrees
    Node* leftSubtree = buildIsolationTree(leftData, leftSize, max_height, current_height + 1, randomFeatures, subsample_size);
    Node* rightSubtree = buildIsolationTree(rightData, rightSize, max_height, current_height + 1, randomFeatures, subsample_size);

    // Free memory allocated for partitioned data
    free(leftData);
    free(rightData);

    // Create and return the current node
    Node* node = createNode(randomFeatureIndex, splitValue);
    node->left = leftSubtree;
    node->right = rightSubtree;
    return node;
}

// Build an ensemble of isolation trees
Node** buildIsolationForest(WeatherData* dataset, size_t dataSize, size_t num_trees, size_t subsample_size, size_t max_height) {
    Node** forest = malloc(num_trees * sizeof(Node*));
    if (!forest) {
        fprintf(stderr, "Failed to allocate memory for isolation forest.\n");
        exit(EXIT_FAILURE);
    }

    // Build each tree in the forest
    for (size_t i = 0; i < num_trees; ++i) {
        // Generate random features for subsampling
        int* randomFeatures = generateRandomFeatures(MAX_FEATURES, subsample_size);
        // Build an isolation tree
        forest[i] = buildIsolationTree(dataset, dataSize, max_height, 0, randomFeatures, subsample_size);
        free(randomFeatures);
    }

    return forest;
}

// Calculate the anomaly score for a data point
float calculateAnomalyScore(Node** forest, size_t num_trees, WeatherData* point, size_t num_features, size_t dataSize) {
    float averagePathLength = 0.0;

    // Calculate the average path length for the point in each tree
    for (size_t i = 0; i < num_trees; ++i) {
        averagePathLength += calculatePathLength(forest[i], point, num_features, 0);
    }

    // Average the path lengths over all trees
    averagePathLength /= num_trees;

    // Calculate the anomaly score
    return pow(2, -averagePathLength / (float)dataSize);
}

// Calculate the path length for a data point in an isolation tree
size_t calculatePathLength(Node* node, WeatherData* point, size_t num_features, size_t current_height) {
    if (node == NULL) {
        return current_height;
    }

    int featureIndex = node->featureIndex;
    float splitValue = node->splitValue;

    if (point->features[featureIndex] <= splitValue) {
        return calculatePathLength(node->left, point, num_features, current_height + 1);
    } else {
        return calculatePathLength(node->right, point, num_features, current_height + 1);
    }
}



// Function to calculate the distance to the k-th nearest neighbor of a point
float kthNearestNeighborDistance(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k) {
    float* distances = malloc(dataSize * sizeof(float));
    if (!distances) {
        fprintf(stderr, "Failed to allocate memory for distances.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate distances from the point to all other points
    for (size_t i = 0; i < dataSize; ++i) {
        distances[i] = euclideanDistance(dataset[i].features, point->features, num_features);
    }

    // Sort distances
    qsort(distances, dataSize, sizeof(float), compareFloats);

    float kthNearestNeighborDist = distances[k - 1];

    free(distances);
    return kthNearestNeighborDist;
}

// Function to calculate the reachability distance between two points
float reachabilityDistance(WeatherData* dataset, size_t dataSize, WeatherData* point1, WeatherData* point2, size_t num_features, int k) {
    float kthNearestDist = kthNearestNeighborDistance(dataset, dataSize, point2, num_features, k);
    float distance = euclideanDistance(point1->features, point2->features, num_features);
    return fmax(distance, kthNearestDist);
}

// Function to calculate the local reachability density of a point
float localReachabilityDensity(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k) {
    float sumReachability = 0.0;
    for (size_t i = 0; i < dataSize; ++i) {
        if (&dataset[i] != point) {
            sumReachability += reachabilityDistance(dataset, dataSize, point, &dataset[i], num_features, k);
        }
    }
    return dataSize / sumReachability;
}

// Function to calculate the local outlier factor (LOF) of a point
float localOutlierFactor(WeatherData* dataset, size_t dataSize, WeatherData* point, size_t num_features, int k) {
    float lrdPoint = localReachabilityDensity(dataset, dataSize, point, num_features, k);

    float sumLrdRatio = 0.0;

    for (size_t i = 0; i < dataSize; ++i) {
        if (&dataset[i] != point) {
            float lrdNeighbor = localReachabilityDensity(dataset, dataSize, &dataset[i], num_features, k);
            float reachDist = reachabilityDistance(dataset, dataSize, point, &dataset[i], num_features, k);
            sumLrdRatio += lrdNeighbor / lrdPoint * reachDist;
        }
    }
    return sumLrdRatio / dataSize;
}
   

// Convert WeatherData struct to svm_node array
// struct svm_node* dataPointToSvmNode(WeatherData* dataPoint) {
//     static struct svm_node node[MAX_FEATURES + 1]; // Extra space for the terminating node

//     for (int i = 0; i < MAX_FEATURES; ++i) {
//         node[i].index = i + 1; // Feature index (1-based)
//         node[i].value = dataPoint->features[i]; // Feature value
//     }
//     node[MAX_FEATURES].index = -1; // Terminate the array with -1 as required by libsvm

//     return node;
// }

// Train OCSVM model
// svm_model* trainOCSVM(WeatherData* dataset, size_t dataSize) {
//     svm_parameter param;
//     param.svm_type = ONE_CLASS; // One-Class SVM
//     param.kernel_type = RBF; // Radial Basis Function (Gaussian) kernel
//     param.gamma = 0.5; // Kernel parameter (gamma)
//     param.nu = 0.1; // nu parameter (controlling the number of support vectors)

//     // Convert data to libsvm format
//     struct svm_problem prob;
//     prob.l = (int)dataSize; // Number of data points
//     prob.y = (double*)malloc(prob.l * sizeof(double));
//     prob.x = (struct svm_node**)malloc(prob.l * sizeof(struct svm_node*));

//     for (size_t i = 0; i < dataSize; ++i) {
//         prob.y[i] = 1; // Labels (all 1 since it's a one-class problem)
//         prob.x[i] = dataPointToSvmNode(&dataset[i]);
//     }

//     // Train model
//     svm_model* model = svm_train(&prob, &param);

//     // Free memory
//     free(prob.y);
//     for (size_t i = 0; i < dataSize; ++i) {
//         free(prob.x[i]);
//     }
//     free(prob.x);

//     return model;
// }

// Predict outliers using trained OCSVM model
// void predictOutliers(svm_model* model, WeatherData* dataset, size_t dataSize) {
//     for (size_t i = 0; i < dataSize; ++i) {
//         struct svm_node* dataPoint = dataPointToSvmNode(&dataset[i]);
//         double result = svm_predict(model, dataPoint);
//         printf("Data point %zu: %s\n", i, result == -1 ? "Outlier" : "Inlier");
//     }
// }



// Compute PCA
// void computePCA(WeatherData* dataset, size_t dataSize, size_t numFeatures) {
//     gsl_matrix* dataMatrix = gsl_matrix_alloc(dataSize, numFeatures);
//     gsl_matrix* covarianceMatrix = gsl_matrix_alloc(numFeatures, numFeatures);
//     gsl_eigen_symmv_workspace* workspace = gsl_eigen_symmv_alloc(numFeatures);

//     // Populate data matrix
//     for (size_t i = 0; i < dataSize; ++i) {
//         for (size_t j = 0; j < numFeatures; ++j) {
//             gsl_matrix_set(dataMatrix, i, j, dataset[i].features[j]);
//         }
//     }

//     // Compute covariance matrix
//     gsl_matrix_set_zero(covarianceMatrix);
//     for (size_t i = 0; i < dataSize; ++i) {
//         for (size_t j = 0; j < dataSize; ++j) {
//             for (size_t k = 0; k < numFeatures; ++k) {
//                 gsl_matrix_set(covarianceMatrix, j, k, gsl_matrix_get(covarianceMatrix, j, k) + gsl_matrix_get(dataMatrix, i, j) * gsl_matrix_get(dataMatrix, i, k));
//             }
//         }
//     }
//     gsl_matrix_scale(covarianceMatrix, 1.0 / (dataSize - 1));

//     // Compute eigenvectors and eigenvalues
//     gsl_vector* eigenvalues = gsl_vector_alloc(numFeatures);
//     gsl_matrix* eigenvectors = gsl_matrix_alloc(numFeatures, numFeatures);
//     gsl_eigen_symmv(covarianceMatrix, eigenvalues, eigenvectors, workspace);
//     gsl_eigen_symmv_sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_ABS_ASC);

//     // Output principal components (eigenvectors)
//     printf("Principal Components (Eigenvectors):\n");
//     for (size_t i = 0; i < numFeatures; ++i) {
//         printf("PC %zu: ", i + 1);
//         for (size_t j = 0; j < numFeatures; ++j) {
//             printf("%f ", gsl_matrix_get(eigenvectors, i, j));
//         }
//         printf("\n");
//     }

//     // Cleanup
//     gsl_matrix_free(dataMatrix);
//     gsl_matrix_free(covarianceMatrix);
//     gsl_vector_free(eigenvalues);
//     gsl_matrix_free(eigenvectors);
//     gsl_eigen_symmv_free(workspace);
// }


void freeIsolationTree(Node* node) {
    if (node == NULL) return;
    freeIsolationTree(node->left);
    freeIsolationTree(node->right);
    free(node);
}


// In your main function
int main() {
    printf("Running program.\n");
    // Load the dataset
    size_t dataSize;
    WeatherData* dataset = readCSV("C:/Users/danie/Desktop/NW2016.csv", &dataSize);

    // Assuming you meant to use the global `numFeatures` for the number of features.
    size_t num_features = numFeatures; // Use global variable numFeatures

    // Define a hypothetical query point (adjust the feature count accordingly)
    float queryPoint[] = { /* feature values here, excluding the class label */ };
    // Calculate based on the queryPoint array size
    num_features = sizeof(queryPoint) / sizeof(queryPoint[0]); 

    // Classify the query point using KNN
    int k = 3; // Starting point for k
    int predictedClass = knnClassify(dataset, dataSize, queryPoint, k, num_features);
    printf("Predicted class for the query point using KNN: %d\n", predictedClass);

    // Calculate ABOD
    float abod = calculateABOD(queryPoint, dataset, dataSize, num_features);
    printf("ABOD value for the query point: %f\n", abod);

    // Calculate the CBLOF score for the query point
    float cblof = calculateCBLOF(dataset, dataSize, queryPoint, k, num_features);
    printf("CBLOF score for the query point: %f\n", cblof);

    // Set hyperparameters
    size_t num_trees = 100;
    size_t subsample_size = 5;
    size_t max_height = ceil(log2(dataSize));

    // Build the isolation forest
    Node** forest = buildIsolationForest(dataset, dataSize, num_trees, subsample_size, max_height);

    // Calculate the anomaly score for the query point
    // Assuming you define another `queryPoint` struct for this part or reuse the one defined earlier
    float anomalyScore = calculateAnomalyScore(forest, num_trees, &dataset[0], num_features, dataSize); // Add dataSize
    printf("Anomaly score for the query point using Isolation Forest: %f\n", anomalyScore);

    for (size_t i = 0; i < num_trees; ++i) {
        freeIsolationTree(forest[i]); // Free memory for each tree
    }
    free(forest);

    // Additional calculations here ...

    // Cleanup: Free memory allocated for the dataset
    free(dataset);

    return 0;
}
