import * as brain from 'brain.js'
import type {
  INeuralNetworkJSON,
  INeuralNetworkOptions,
  INeuralNetworkTrainOptions,
} from 'brain.js/dist/src/neural-network'
import { Inputs } from './ga'
import { MirrorNetSpec } from './nn'

export type BrainSpec = {
  netSpec: MirrorNetSpec
  gpu?: boolean
}

export type BrainTrainSpec = {
  dataset: Inputs[]
} & Partial<INeuralNetworkTrainOptions>

export type BrainNet =
  | brain.NeuralNetwork<any, any>
  | brain.NeuralNetworkGPU<any, any>

export function createBrain(spec: BrainSpec) {
  const { netSpec } = spec
  const { layers } = netSpec
  const hiddenLayers: number[] = []

  const L = layers.length
  let l: number
  for (l = 1; l < L; l++) {
    hiddenLayers.push(layers[l])
  }
  for (l = L - 2; l > 0; l--) {
    hiddenLayers.push(layers[l])
  }

  const options: Partial<INeuralNetworkOptions & INeuralNetworkTrainOptions> = {
    inputSize: layers[0],
    hiddenLayers,
    outputSize: layers[0],
    activation: 'tanh',
  }

  const net = spec.gpu
    ? new brain.NeuralNetworkGPU<Inputs, Float32Array>({
        ...options,
        mode: 'gpu',
      })
    : new brain.NeuralNetwork<Inputs, Float32Array>(options)
  function train(options: BrainTrainSpec) {
    const data = options.dataset.map((xs) => ({
      input: xs,
      output: xs as Float32Array,
    }))
    return net.train(data, options)
  }
  return { net, train }
}

export function splitBrain(net: BrainNet) {
  let json: INeuralNetworkJSON = net.toJSON()
  const { sizes, layers } = json
  const L = sizes.length
  if (L % 2 !== 1) {
    throw new Error(
      'invalid network, the layer structure is not mirrored, expect odd number of layers'
    )
  }
  const mid = (L - 1) / 2
  const encoderSize = sizes.slice(0, mid + 1)
  const decoderSize = sizes.slice(mid)

  let i = 0
  let j = mid
  for (; j >= 0; ) {
    if (encoderSize[i] !== decoderSize[j]) {
      throw new Error(
        `invalid network, the layer structure is not mirrored. encoderSize[${i}] is ${encoderSize[i]}, decoderSize[${j}] is ${decoderSize[j]}. they should be the same`
      )
    }
    i++
    j--
  }

  const encoderLayers = layers.slice(0, L - mid)
  const decoderLayers = layers.slice(mid)
  decoderLayers[0] = {
    weights: [],
    biases: [],
  }

  const encoderJSON: INeuralNetworkJSON = {
    ...json,
    sizes: encoderSize,
    layers: encoderLayers,
  }
  const decoderJSON: INeuralNetworkJSON = {
    ...json,
    sizes: decoderSize,
    layers: decoderLayers,
  }

  if (json.type === 'NeuralNetwork') {
    const encoder = new brain.NeuralNetwork<Inputs, Float32Array>()
    encoder.fromJSON(encoderJSON)
    const decoder = new brain.NeuralNetwork<Inputs, Float32Array>()
    decoder.fromJSON(decoderJSON)
    return { encoder, decoder }
  } else if (json.type === 'NeuralNetworkGPU') {
    const encoder = new brain.NeuralNetworkGPU<Inputs, Float32Array>()
    encoder.fromJSON(encoderJSON)
    const decoder = new brain.NeuralNetworkGPU<Inputs, Float32Array>()
    decoder.fromJSON(decoderJSON)
    return { encoder, decoder }
  } else {
    throw new Error('unknown tyoe of network: ' + json.type)
  }
}
