import { tanh } from './math'

const { random } = Math

export function randomWeight() {
  const r = random() + random()
  return r > 1 ? 2 - r : -r
}

export function randomBias() {
  return random() * 2 - 1
}

export type MirrorNet = {
  w: Float32Array
  b: Float32Array
  o: Float32Array[]
}

export type MirrorNetSpec = {
  layers: number[]
}

/**
 * e.g. [2,3] -> [2x3] -> [2x3 | 3x2]
 * e.g. [8,4,2] -> [8x4,4x2] -> [8x4, 4x2 | 2x4, 4x8]
 * */
export function randomMirrorNet(spec: MirrorNetSpec): MirrorNet {
  const w: number[] = []
  const b: number[] = []
  const o: Float32Array[] = []

  const { layers } = spec
  const L = layers.length
  let inputSize = layers[0]
  let outputSize: number
  let l: number
  let y: number
  let x: number

  /* decoder output layer */
  for (y = 0; y < inputSize; y++) {
    b.push(randomBias())
  }
  o.push(new Float32Array(inputSize))

  /* encoder layers */
  for (l = 1; l < L; l++) {
    outputSize = layers[l]
    for (y = 0; y < outputSize; y++) {
      for (x = 0; x < inputSize; x++) {
        w.push(randomWeight())
      }
      b.push(randomBias())
    }
    o.push(new Float32Array(outputSize))
    inputSize = outputSize
  }

  return {
    w: new Float32Array(w),
    b: new Float32Array(b),
    o,
  }
}

export function runMirrorNet(
  net: MirrorNet,
  inputs: number[] | Float32Array
): Float32Array {
  const { w, b, o } = net
  const L = o.length
  let inputSize = inputs.length
  let outputSize: number
  let outputs: Float32Array
  let l: number
  let y: number
  let x: number
  let wi = 0
  let bi = inputSize
  let acc: number

  /* encode */
  for (l = 1; l < L; l++) {
    /* init */
    outputs = o[l]
    outputSize = outputs.length
    /* calc */
    for (y = 0; y < outputSize; y++) {
      /* calc sum of x*w */
      acc = 0
      for (x = 0; x < inputSize; x++) {
        acc += inputs[x] * w[wi]
        wi++
      }
      /* add bias, then run activation */
      outputs[y] = tanh(acc + b[bi])
      bi++
    }
    /* pass to next iteration */
    inputSize = outputSize
    inputs = outputs
  }
  bi -= outputSize!

  /* decode */
  for (l = L - 2; l >= 0; l--) {
    /* init */
    outputs = o[l]
    outputSize = outputs.length
    for (y = outputSize - 1; y >= 0; y--) {
      outputs[y] = 0
    }
    /* calc */
    for (x = inputSize - 1; x >= 0; x--) {
      /* calc sum of x*w */
      for (y = outputSize - 1; y >= 0; y--) {
        wi--
        outputs[y] += inputs[x] * w[wi]
      }
    }
    /* add bias, then run activation */
    for (y = outputSize - 1; y >= 0; y--) {
      bi--
      outputs[y] = tanh(outputs[y] + b[bi])
    }
    /* pass to next iteration */
    inputSize = outputSize
    inputs = outputs
  }

  return inputs as Float32Array
}

export function mirrorEncode(
  net: MirrorNet,
  inputs: number[] | Float32Array
): Float32Array {
  const { w, b, o } = net
  const L = o.length
  let inputSize = inputs.length
  let outputSize: number
  let outputs: Float32Array
  let l: number
  let y: number
  let x: number
  let wi = 0
  let bi = inputSize
  let acc: number

  /* encode */
  for (l = 1; l < L; l++) {
    /* init */
    outputs = o[l]
    outputSize = outputs.length
    /* calc */
    for (y = 0; y < outputSize; y++) {
      /* calc sum of x*w */
      acc = 0
      for (x = 0; x < inputSize; x++) {
        acc += inputs[x] * w[wi]
        wi++
      }
      /* add bias, then run activation */
      outputs[y] = tanh(acc + b[bi])
      bi++
    }
    /* pass to next iteration */
    inputSize = outputSize
    inputs = outputs
  }

  return inputs as Float32Array
}
export function mirrorDecode(
  net: MirrorNet,
  inputs: number[] | Float32Array
): Float32Array {
  const { w, b, o } = net
  const L = o.length
  let inputSize = inputs.length
  let outputSize: number
  let outputs: Float32Array
  let l: number
  let y: number
  let x: number
  let wi = w.length
  let bi = b.length - o[L - 1].length

  /* decode */
  for (l = L - 2; l >= 0; l--) {
    /* init */
    outputs = o[l]
    outputSize = outputs.length
    for (y = outputSize - 1; y >= 0; y--) {
      outputs[y] = 0
    }
    /* calc */
    for (x = inputSize - 1; x >= 0; x--) {
      /* calc sum of x*w */
      for (y = outputSize - 1; y >= 0; y--) {
        wi--
        outputs[y] += inputs[x] * w[wi]
      }
    }
    /* add bias, then run activation */
    for (y = outputSize - 1; y >= 0; y--) {
      bi--
      outputs[y] = tanh(outputs[y] + b[bi])
    }
    /* pass to next iteration */
    inputSize = outputSize
    inputs = outputs
  }

  return inputs as Float32Array
}
