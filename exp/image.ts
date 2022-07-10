import { best, GaIsland } from 'ga-island'
import brain from 'brain.js'
import { selectImage } from '@beenotung/tslib/file'
import type { INeuralNetworkJSON } from 'brain.js/dist/src/neural-network'
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
let zoomRate = 1

const canvas = querySelector<HTMLCanvasElement>('canvas#allrgb')
const loadImageButton = querySelector<HTMLButtonElement>('#load-image')
const acticationButton = querySelector<HTMLButtonElement>('#activation-fn')
const statusButton = querySelector<HTMLButtonElement>('#running-status')
const inputSizeSpan = querySelector('#input-size')
const hiddenLayersInput = querySelector<HTMLInputElement>('#hidden-layers')
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

const inputSize = expandInput(0, 0).length
const hiddenLayers = hiddenLayersInput.value || '64,9'
const outputSize = 3

inputSizeSpan.textContent = String(inputSize)
if (!hiddenLayersInput.value) {
  hiddenLayersInput.value = hiddenLayers
}

let encoder = new brain.NeuralNetwork({
  inputSize,
  hiddenLayers: hiddenLayers.split(',').map(s => parseInt(s)),
  outputSize,
  iterations: 10,
  activation,
  learningRate,
})
let json: INeuralNetworkJSON | null = encoder.toJSON()
json = null
if (json) {
  encoder.fromJSON(json)
}

let running = false

statusButton.textContent = 'running: ' + running
canvas.onclick = () => {
  running = !running
  statusButton.textContent = 'running: ' + running
  if (running) {
    setTimeout(loop)
  }
}
statusButton.onclick = canvas.onclick
loadImageButton.onclick = () => {
  loadImage()
}
acticationButton.textContent = activation
acticationButton.onclick = () => {
  location.search =
    '?activation=' + (activation === 'tanh' ? 'sigmoid' : 'tanh')
}
hiddenLayersInput.onchange = () => {
  encoder = new brain.NeuralNetwork({
    inputSize,
    hiddenLayers: hiddenLayers.split(',').map(s => parseInt(s)),
    outputSize,
    iterations: 10,
    activation,
    learningRate,
  })
}
function calcInputWidth() {
  hiddenLayersInput.style.width = hiddenLayersInput.value.length + 1 + 'ch'
}
hiddenLayersInput.oninput = calcInputWidth
calcInputWidth()

function expandInput(x: number, y: number) {
  let output: number[] = []
  for (let i of [(x / w) * 2 - 1, (y / h) * 2 - 1]) {
    i *= zoomRate
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

function setZoomRate(_zoomRate?: number) {
  if (!_zoomRate) {
    return zoomRate
  }
  zoomRate = _zoomRate
  if (json) {
    running = true
    loop()
  }
}

Object.assign(window, {
  encoder,
  loadImage,
  imageData,
  expandInput,
  zoomRate: setZoomRate,
})

let gen = 0

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

function querySelector<T extends HTMLElement>(selector: string) {
  let e = document.querySelector(selector)
  if (!e) throw new Error('Element not found, selector: ' + selector)
  return e as T
}
