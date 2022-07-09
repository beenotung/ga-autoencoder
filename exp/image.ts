import { best, GaIsland } from 'ga-island'
import brain from 'brain.js'
import { selectImage } from '@beenotung/tslib/file'
import type { INeuralNetworkJSON } from 'brain.js/dist/src/neural-network';
import {
  base64ToCanvas,
  base64ToImage,
  compressMobilePhoto,
} from '@beenotung/tslib/image'

const { sin, cos, tan, sqrt, round, floor, ceil, random, min, max } = Math

let R = 0
let G = 1
let B = 2
let A = 3

let scale = 5
let step = 3

const canvas: HTMLCanvasElement = document.querySelector('canvas#allrgb')!
const loadImageButton: HTMLButtonElement =
  document.querySelector('#load-image')!
const rect = canvas.getBoundingClientRect()
const w = floor(rect.width / scale)
const h = floor(rect.height / scale)
canvas.width = w
canvas.height = h
console.log({ w, h })
const context = canvas.getContext('2d')!
const imageData = context.createImageData(w, h)

let activation = 'sigmoid'
let learningRate = 0.3

if (location.search === '?activation=tanh') {
  activation = 'tanh'
  learningRate = 0.01
}

const encoder = new brain.NeuralNetwork({
  inputSize: expandInput(0, 0).length,
  hiddenLayers: [64, 9],
  outputSize: 3,
  iterations: 10,
  activation,
  learningRate,
})
let json: INeuralNetworkJSON | null = encoder.toJSON()
json = null
if (json) {
  encoder.fromJSON(json)
}

canvas.onclick = () => {
  running = !running
  if (running) {
    loop()
  }
}
loadImageButton.onclick = () => {
  loadImage()
}

function expandInput(x: number, y: number) {
  let output: number[] = []
  for (let i of [(x / w) * 2 - 1, (y / h) * 2 - 1]) {
    for (let x of [i, i ** 2, i ** 3, i ** 4]) {
      output.push(x, sin(x), cos(x), tan(x))
    }
  }
  return output
}

const encodeOutput =
  activation === 'tanh'
    ? function encodeOutput(
        r: number,
        g: number,
        b: number,
      ): [number, number, number] {
        return [r, g, b].map(x => (x / 255) * 2 - 1) as any
      }
    : function encodeOutput(
        r: number,
        g: number,
        b: number,
      ): [number, number, number] {
        return [r, g, b].map(x => x / 255) as any
      }
const decodeOutput =
  activation === 'tanh'
    ? function decodeOutput(
        output: Float32Array | number[],
      ): [number, number, number] {
        return output.map(x => floor(((x + 1) / 2) * 255)) as any
      }
    : function decodeOutput(
        output: Float32Array | number[],
      ): [number, number, number] {
        return output.map(x => floor(x * 255)) as any
      }

async function loadImage() {
  let files = await selectImage()
  let file = files[0]
  if (!file) return
  let dataUrl = await compressMobilePhoto({ image: file })
  if (!'dev') {
    let image = await base64ToImage(dataUrl)
    context.drawImage(image, 0, 0, w, h)
  } else {
    let canvas = await base64ToCanvas(dataUrl)
    let context = canvas.getContext('2d')!
    let { width, height } = canvas
    let imageData = context.getImageData(0, 0, width, height)
    for (let y = 0, i = 0; y < h; y += step) {
      for (let x = 0; x < w; x += step, i++) {
        let o = (floor((y / h) * height) * width + floor((x / w) * width)) * 4
        encoderDataset[i] = {
          input: expandInput(x, y),
          output: encodeOutput(
            imageData.data[o + 0],
            imageData.data[o + 1],
            imageData.data[o + 2],
          ),
        }
      }
    }
  }
}

Object.assign(window, { encoder, loadImage, imageData, expandInput })

let gen = 0
let running = false

let encoderDataset: {
  input: number[]
  output: [r: number, g: number, b: number]
}[] = []
for (let y = 0, i = 0; y < h; y += step) {
  for (let x = 0; x < w; x += step, i++) {
    let r = Math.random() * 255
    let g = Math.random() * 255
    let b = Math.random() * 255
    encoderDataset[i] = {
      input: expandInput(x, y),
      output: encodeOutput(r, g, b),
    }
  }
}

function loop() {
  if (!running) return
  if (json) {
    running = false
  } else {
    gen++
    encoder.train(encoderDataset)
  }

  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i += 4) {
      let output = encoder.run(expandInput(x, y)) as number[]
      let res = decodeOutput(output)
      imageData.data[i + R] = res[0]
      imageData.data[i + G] = res[1]
      imageData.data[i + B] = res[2]
      imageData.data[i + A] = 255
    }
  }

  context.putImageData(imageData, 0, 0)
  requestAnimationFrame(loop)
}
loop()
