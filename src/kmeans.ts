import { assertUnreachable, validateKmeansInput } from './assert.js';
import { calculateDistanceMatrix, hasConverged, updateCenters, updateClusterID } from './utils.js';
import { kmeanspp, mostDistant, random } from './initialization.js';

import { KMeansResult } from './KMeansResult.js';
import { squaredEuclidean } from 'ml-distance-euclidean';

const defaultOptions = { maxIterations: 100, tolerance: 1e-6, initialization: 'kmeans++' as InitializationMethod, distanceFunction: squaredEuclidean };

export type InitializationMethod = 'kmeans++' | 'random' | 'mostDistant';
export type Options = OptionsWithDefault & OptionsWithoutDefault;
type DefinedOptions = Required<OptionsWithDefault> & OptionsWithoutDefault;

export interface OptionsWithDefault {
  distanceFunction?: (p: number[], q: number[]) => number;
  tolerance?: number;
  initialization?: InitializationMethod | number[][];
  maxIterations?: number;
}
export interface OptionsWithoutDefault {
  seed?: number;
}

function step(centers: number[][], data: number[][], clusterID: number[], K: number, options: DefinedOptions, iterations: number): KMeansResult {
  clusterID = updateClusterID(data, centers, clusterID, options.distanceFunction);
  const newCenters = updateCenters(centers, data, clusterID, K);
  return new KMeansResult(clusterID, newCenters, hasConverged(newCenters, centers, options.distanceFunction, options.tolerance), iterations, options.distanceFunction);
}
export function* kmeansGenerator(data: number[][], K: number, options: Options) {
  const definedOptions = getDefinedOptions(options);
  validateKmeansInput(data, K);
  let centers = initializeCenters(data, K, definedOptions), converged = false, stepResult: KMeansResult;
  for (let stepNumber = 0; !converged && stepNumber < definedOptions.maxIterations; stepNumber++) {
    stepResult = step(centers, data, new Array(data.length), K, definedOptions, stepNumber);
    yield stepResult;
    converged = stepResult.converged;
    centers = stepResult.centroids;
  }
}
export function kmeans(data: number[][], K: number, options: Options) {
  const definedOptions = getDefinedOptions(options);
  validateKmeansInput(data, K);
  let centers = initializeCenters(data, K, definedOptions);
  if (definedOptions.maxIterations === 0) definedOptions.maxIterations = Number.MAX_VALUE;
  let converged = false, stepResult: KMeansResult;
  for (let stepNumber = 0; !converged && stepNumber < definedOptions.maxIterations; stepNumber++) {
    stepResult = step(centers, data, new Array(data.length), K, definedOptions, stepNumber);
    converged = stepResult.converged;
    centers = stepResult.centroids;
  }
  if (!stepResult) throw new Error('unreachable: no kmeans step executed');
  return stepResult;
}
function initializeCenters(data: number[][], K: number, options: DefinedOptions) {
  let centers: number[][];
  if (Array.isArray(options.initialization)) {
    if (options.initialization.length !== K) throw new Error('The initial centers should have the same length as K');
    else centers = options.initialization;
  } else switch (options.initialization) {
    case 'kmeans++': centers = kmeanspp(data, K, options); break;
    case 'random': centers = random(data, K, options.seed); break;
    case 'mostDistant': centers = mostDistant(data, K, calculateDistanceMatrix(data, options.distanceFunction), options.seed); break;
    default: assertUnreachable(options.initialization, 'Unknown initialization method');
  }
  return centers;
}
function getDefinedOptions(options: Options): DefinedOptions {
  return { ...defaultOptions, ...options };
}
