import { best, GaIsland } from 'ga-island'

let { exp, log, E, random } = Math

type Gene = {
  w: number[]
  b: number[]
  y: number[]
}

let input = [1, 0.5]

function sigmoid(x: number): number {
  return 1 / (1 + exp(-x))
}

let log_e = log(E)
function reverse_sigmoid(y: number): number {
  return log(y / (1 - y)) / log_e
}

let mutate_rate = 0.25
let ga = new GaIsland<Gene>({
  populationSize: 16,
  mutationRate: 20,
  crossover(aParent, bParent, child) {
    for (let i = 0; i < 4; i++) {
      child.w[i] = random() < 0.5 ? aParent.w[i] : bParent.w[i]
    }
    for (let i = 0; i < 3; i++) {
      child.b[i] = random() < 0.5 ? aParent.b[i] : bParent.b[i]
    }
  },
  mutate(input, output) {
    for (let i = 0; i < 4; i++) {
      output.w[i] =
        input.w[i] + (random() < mutate_rate ? (random() * 2 - 1) * 0.25 : 0)
    }
    for (let i = 0; i < 3; i++) {
      output.b[i] =
        input.b[i] + (random() < mutate_rate ? (random() * 2 - 1) * 0.25 : 0)
    }
  },
  fitness(gene) {
    run(gene)
    let y = gene.y
    let d_0 = y[0] - input[0]
    let d_1 = y[1] - input[1]
    return 1 / (d_0 * d_0 + d_1 * d_1 + 1e-5)
  },
  randomIndividual() {
    return {
      w: [random(), random(), random(), random()],
      b: [random(), random(), random()],
      y: [0, 0, 0],
    }
  },
})

function run(gene: Gene) {
  let w = gene.w
  let b = gene.b
  let y = gene.y
  let value = sigmoid(input[0] * w[0] + input[1] * w[1] + b[0])
  y[0] = sigmoid(value * w[2] + b[1])
  y[1] = sigmoid(value * w[3] + b[2])
}

Object.assign(window, { ga, run, best })

for (let i = 1; i <= 200; i++) {
  ga.evolve()
  let { fitness, gene } = best(ga.options)
  run(gene)
  console.log(i, fitness, gene.y, gene)
}
