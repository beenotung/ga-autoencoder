import brain from 'brain.js'

let net = new brain.NeuralNetwork({
  inputSize: 2,
  hiddenLayers: [1],
  outputSize: 2,
  activation: 'sigmoid',
})

let input = [1, 0.5]
let state = net.train([{ input: input, output: input }])

console.log(state)

Object.assign(window, {
  state,
  net,
  brain,
})
