import { GaIsland } from 'ga-island'
import { MirrorNet, MirrorNetSpec, randomMirrorNet, runMirrorNet } from './nn'

const { random, sqrt } = Math

export type GaSpec = {
  netSpec: MirrorNetSpec
}

export type Inputs = number[] | Float32Array

export type GATrainSpec = {
  dataset: Array<Inputs>
  iterations?: number // default to 1
}

export function createGaPool(spec: GaSpec) {
  const { netSpec } = spec
  const Dim = netSpec.layers[0]
  const mutateAmount = 0.25

  let dataset: Array<Inputs>
  let N = 0

  const ga = new GaIsland<MirrorNet>({
    mutationRate: 0.2,
    populationSize: 16,
    randomIndividual: () => randomMirrorNet(netSpec),
    crossover: (a, b, c) => {
      const aw = a.w
      const ab = a.b

      const bw = b.w
      const bb = b.b

      const cw = c.w
      const cb = c.b

      const W = aw.length
      const B = ab.length

      for (let i = 0; i < W; i++) {
        cw[i] = random() < 0.5 ? aw[i] : bw[i]
      }
      for (let i = 0; i < B; i++) {
        cb[i] = random() < 0.5 ? ab[i] : bb[i]
      }
    },
    mutate: (a, b) => {
      const aw = a.w
      const ab = a.b

      const bw = b.w
      const bb = b.b

      const W = aw.length
      const B = ab.length

      let i: number
      let r: number
      for (i = 0; i < W; i++) {
        r = random()
        bw[i] = aw[i] + (r < mutateAmount ? r * 2 - 1 : 0)
      }
      for (i = 0; i < B; i++) {
        r = random()
        bb[i] = ab[i] + (r < mutateAmount ? r * 2 - 1 : 0)
      }
    },
    fitness: (net) => {
      let fitness = 0
      let acc = 0
      let i: number
      let dim: number
      let inputs: Inputs
      let outputs: Inputs
      let e: number
      for (i = 0; i < N; i++) {
        inputs = dataset[i]
        outputs = runMirrorNet(net, inputs)
        acc = 0
        for (dim = 0; dim < Dim; dim++) {
          e = inputs[dim] - outputs[dim]
          acc += e * e
        }
        fitness -= sqrt(acc)
      }
      return fitness
    },
  })
  function setDataset(ds: Array<Inputs>) {
    dataset = ds
    N = dataset.length
  }
  function train(options: GATrainSpec) {
    setDataset(options.dataset)
    const iteration = options.iterations || 1
    let i: number
    for (i = 0; i < iteration; i++) {
      ga.evolve()
    }
  }
  return { ga, setDataset, train }
}
