'use strict';

/**
 * Calculates the sum of squared errors
 * @ignore
 * @param {Array<Array<Number>>} data - the [x,y,z,...] points to cluster
 * @param {Array<Array<Number>>} centers - the K centers in format [x,y,z,...]
 * @param {Array<Number>} clusterID - the cluster identifier for each data dot
 * @param {Function} distance - Distance function to use between the points
 * @returns {Array<Number>} the sum of squared errors
 */
function computeDispersion(data, centers, clusterID, distance) {
    var sse = new Array(centers.length);
    var sseLen = new Array(centers.length);
    for (var s = 0; s < centers.length; s++) {
        sse[s] = 0;
        sseLen[s] = 0;
    }

    for (var i = 0; i < data.length; i++) {
        sse[clusterID[i]] += distance(data[i], centers[clusterID[i]]);
        sseLen[clusterID[i]]++;
    }

    for (var j = 0; j < centers.length; j++) {
        sse[j] /= sseLen[j];
    }
    return sse;
}

/**
 * Calculates the distance matrix for a given array of points
 * @ignore
 * @param {Array<Array<Number>>} data - the [x,y,z,...] points to cluster
 * @param {Function} distance - Distance function to use between the points
 * @return {Array<Array<Number>>} - matrix with the distance values
 */
function calculateDistanceMatrix(data, distance) {
    var distanceMatrix = new Array(data.length);
    for (var i = 0; i < data.length; ++i) {
        for (var j = i; j < data.length; ++j) {
            if (!distanceMatrix[i]) {
                distanceMatrix[i] = new Array(data.length);
            }
            if (!distanceMatrix[j]) {
                distanceMatrix[j] = new Array(data.length);
            }
            const dist = distance(data[i], data[j]);
            distanceMatrix[i][j] = dist;
            distanceMatrix[j][i] = dist;
        }
    }
    return distanceMatrix;
}

/**
 * Updates the cluster identifier based in the new data
 * @ignore
 * @param {Array<Array<Number>>} data - the [x,y,z,...] points to cluster
 * @param {Array<Array<Number>>} centers - the K centers in format [x,y,z,...]
 * @param {Function} distance - Distance function to use between the points
 * @returns {Array} the cluster identifier for each data dot
 */
function updateClusterID(data, centers, distance) {
    const k = centers.length;
    var clusterID = new Array(data.length);
    for (var index = 0; index < data.length; index++)
        clusterID[index] = 0;

    var aux = 0;
    var distance2Centroid = new Array(data.length);
    for (var i = 0; i < data.length; i++) {
        distance2Centroid[i] = new Array(k);
        for (var j = 0; j < k; j++) {
            aux = distance(data[i], centers[j]);
            distance2Centroid[i][j] = [aux, j];
        }
        var min = distance2Centroid[i][0][0];
        var id = 0;
        for (var l = 0; l < k; l++) {
            if (distance2Centroid[i][l][0] < min) {
                min  = distance2Centroid[i][l][0];
                id = distance2Centroid[i][l][1];
            }
        }
        clusterID[i] = id;
    }
    return clusterID;
}

/**
 * Update the center values based in the new configurations of the clusters
 * @ignore
 * @param {Array <Array <Number>>} data - the [x,y,z,...] points to cluster
 * @param {Array <Number>} clusterID - the cluster identifier for each data dot
 * @param {Number} K - Number of clusters
 * @returns {Array} he K centers in format [x,y,z,...]
 */
function updateCenters(data, clusterID, K) {
    const nDim = data[0].length;

    // creates empty centers with 0 size
    var centers = new Array(K);
    var centersLen = new Array(K);
    for (var i = 0; i < K; i++) {
        centers[i] = new Array(nDim);
        centersLen[i] = 0;
        for (var j = 0; j < nDim; j++) {
            centers[i][j] = 0;
        }
    }

    // add the value for all dimensions of the point
    for (var l = 0; l < data.length; l++) {
        centersLen[clusterID[l]]++;
        for (var dim = 0; dim < nDim; dim++) {
            centers[clusterID[l]][dim] += data[l][dim];
        }
    }

    // divides by length
    for (var id = 0; id < K; id++) {
        for (var d = 0; d < nDim; d++) {
            centers[id][d] /= centersLen[id];
        }
    }
    return centers;
}

/**
 * The centers have moved more than the tolerance value?
 * @ignore
 * @param {Array<Array<Number>>} centers - the K centers in format [x,y,z,...]
 * @param {Array<Array<Number>>} oldCenters - the K old centers in format [x,y,z,...]
 * @param {Function} distanceFunction - Distance function to use between the points
 * @param {Number} tolerance - Allowed distance for the centroids to move
 * @return {boolean}
 */
function converged(centers, oldCenters, distanceFunction, tolerance) {
    for (var i = 0; i < centers.length; i++) {
        if (distanceFunction(centers[i], oldCenters[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

exports.computeDispersion = computeDispersion;
exports.updateClusterID = updateClusterID;
exports.updateCenters = updateCenters;
exports.calculateDistanceMatrix = calculateDistanceMatrix;
exports.converged = converged;
