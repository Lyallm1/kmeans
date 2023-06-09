import { updateClusterID } from './utils.js';

export interface CentroidWithInformation {
  centroid: number[];
  error: number;
  size: number;
}

export class KMeansResult {
  constructor(public clusters: number[], public centroids: number[][], public converged: boolean, public iterations: number, public distance: (a: number[], b: number[]) => number) {}

  nearest(data: number[][]): number[] {
    return updateClusterID(data, this.centroids, new Array(data.length), this.distance);
  }

  computeInformation(data: number[][]): CentroidWithInformation[] {
    const enrichedCentroids = this.centroids.map(centroid => ({ centroid, error: 0, size: 0 }));
    data.forEach((a, i) => {
      enrichedCentroids[this.clusters[i]].error += this.distance(a, this.centroids[this.clusters[i]]);
      enrichedCentroids[this.clusters[i]].size++;
    });
    this.centroids.forEach((_, j) => {
      let error = enrichedCentroids[j].error;
      if (enrichedCentroids[j].size && error !== -1) error /= enrichedCentroids[j].size;
      else enrichedCentroids[j].error = -1;
    });
    return enrichedCentroids;
  }
}
