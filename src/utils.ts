import nearestVector from 'ml-nearest-vector';

export function calculateDistanceMatrix(data: number[][], distance: (a: number[], b: number[]) => number) {
  const distanceMatrix = data;
  for (let [i, d] of data.entries()) for (let j = i; j < data.length; ++j) {
    if (!distanceMatrix[i]) distanceMatrix[i] = new Array(data.length);
    if (!distanceMatrix[j]) distanceMatrix[j] = new Array(data.length);
    distanceMatrix[i][j] = distanceMatrix[j][i] = distance(d, data[j]);
  }
  return distanceMatrix;
}
export function updateClusterID(data: number[][], centers: number[][], clusterID: number[], distance: (a: number[], b: number[]) => number): number[] {
  data.forEach((d, i) => clusterID[i] = nearestVector(centers, d, { distanceFunction: distance }));
  return clusterID;
}
export function updateCenters(prevCenters: number[][], data: number[][], clusterID: number[], K: number,): number[][] {
  const nDim = data[0].length, centers = new Array<number[]>(K), centersLen = new Array<number>(K);
  for (let i = 0; i < K; i++) {
    centers[i] = new Array<number>(nDim).fill(0);
    centersLen[i] = 0;
  }
  data.forEach((d, l) => {
    centersLen[clusterID[l]]++;
    for (let dim = 0; dim < nDim; dim++) centers[clusterID[l]][dim] += d[dim];
  });
  for (let id = 0; id < K; id++) for (let d = 0; d < nDim; d++) centers[id][d] = centersLen[id] ? centers[id][d] / centersLen[id] : prevCenters[id][d];
  return centers;
}
export function hasConverged(centers: number[][], oldCenters: number[][], distanceFunction: (a: number[], b: number[]) => number, tolerance: number): boolean {
  for (const [i, center] of centers.entries()) if (distanceFunction(center, oldCenters[i]) > tolerance) return false;
  return true;
}
