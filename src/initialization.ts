import { Matrix } from 'ml-matrix';
import Random from 'ml-random';
import { squaredEuclidean } from 'ml-distance-euclidean';

export function random(data: number[][], K: number, seed?: number) {
  return new Random(seed).choice(data, { size: K });
}
export function mostDistant(data: number[][], K: number, distanceMatrix: number[][], seed?: number): number[][] {
  const random = new Random(seed), ans = new Array<number>(K);
  ans[0] = Math.floor(random.random() * data.length);
  if (K > 1) {
    let maxDist = { dist: -1, index: -1 };
    for (const l in data) if (distanceMatrix[ans[0]][l] > maxDist.dist) [maxDist.dist, maxDist.index] = [distanceMatrix[ans[0]][l], Number(l)];
    ans[1] = maxDist.index;
    if (K > 2) for (let k = 2; k < K; ++k) {
      let center = { dist: -1, index: -1 };
      for (const m in data) {
        let minDistCent = { dist: Number.MAX_VALUE, index: -1 };
        for (let n = 0; n < k; ++n) if (distanceMatrix[n][m] < minDistCent.dist && !ans.includes(Number(m))) minDistCent = { dist: distanceMatrix[n][m], index: Number(m) };
        if (minDistCent.dist !== Number.MAX_VALUE && minDistCent.dist > center.dist) center = { ...minDistCent };
      }
      ans[k] = center.index;
    }
  }
  return ans.map(index => data[index]);
}
export function kmeanspp(X: number[][], K: number, options: Partial<{ seed: number, localTrials: number }> = {}) {
  const m = new Matrix(X), nSamples = m.rows, random = new Random(options.seed), centers: number[][] = [],
  localTrials = options.localTrials || 2 + Math.floor(Math.log(K));
  centers.push(m.getRow(random.randInt(nSamples)));
  let closestDistSquared = new Matrix(1, m.rows);
  for (let i = 0; i < m.rows; i++) closestDistSquared.set(0, i, squaredEuclidean(m.getRow(i), centers[0]));
  let cumSumClosestDistSquared = [cumSum(closestDistSquared.getRow(0))], probabilities = Matrix.mul(closestDistSquared, 1 / cumSumClosestDistSquared[0][nSamples - 1]);
  for (let i = 1; i < K; i++) {
    const candidateIdx = random.choice(nSamples, { replace: true, size: localTrials, probabilities: probabilities.getRow(0) });
    let bestCandidate = Infinity, bestPot = Infinity, bestDistSquared = closestDistSquared;
    for (let j = 0; j < localTrials; j++) {
      const newDistSquared = Matrix.min(closestDistSquared, [euclideanDistances(m.selection(candidateIdx, range(m.columns)), m).getRow(j)]), newPot = newDistSquared.sum();
      if (newPot < bestPot) [bestCandidate, bestPot, bestDistSquared] = [candidateIdx[j], newPot, newDistSquared];
    }
    centers[i] = m.getRow(bestCandidate);
    closestDistSquared = bestDistSquared;
    cumSumClosestDistSquared = [cumSum(closestDistSquared.getRow(0))];
    probabilities = Matrix.mul(closestDistSquared, 1 / cumSumClosestDistSquared[0][nSamples - 1]);
  }
  return centers;
}
function euclideanDistances(A: Matrix, B: Matrix) {
  const result = new Matrix(A.rows, B.rows);
  for (let i = 0; i < A.rows; i++) for (let j = 0; j < B.rows; j++) result.set(i, j, squaredEuclidean(A.getRow(i), B.getRow(j)));
  return result;
}
function range(l: number): number[] {
  return Array.from({ length: l }, (_, i) => i);
}
function cumSum(arr: number[]): number[] {
  const cumSum = [arr[0]];
  for (let i = 1; i < arr.length; i++) cumSum[i] = cumSum[i - 1] + arr[i];
  return cumSum;
}
