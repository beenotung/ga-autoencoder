import { expect } from 'chai'
import { best } from 'ga-island'
import { createGaPool } from './ga'
import { decode, encode } from './nn'

const { sqrt, random } = Math

describe('fitness function', () => {
  it('should consider global distance', () => {
    let e1 = [0.8, 0.8, 0.2, 0.2]
    let e2 = [0.8, 0.2, 0.8, 0.2]
    let sum1 = e1[0] + e1[1] + e1[2] + e1[3]
    let sum2 = e2[0] + e2[1] + e2[2] + e2[3]
    let ss1 = sqrt(e1[0] + e1[1]) + sqrt(e1[2] + e1[3])
    let ss2 = sqrt(e2[0] + e2[1]) + sqrt(e2[2] + e2[3])
    // console.log({ sum1, sum2, ss1, ss2 })
    expect(ss1).lessThan(sum1)
    expect(ss2).closeTo(sum2, 1e-7)
  })
})

describe('run training on sample', () => {
  it('should coverage', () => {
    const dataset: Float32Array[] = []
    const N = 100
    const Round = 10_000_000
    for (let i = 0; i < N; i++) {
      let data = new Float32Array(2)
      data[0] = random() * 2 - 1
      data[1] = random() * 2 - 1
      dataset.push(data)
    }
    const pool = createGaPool({
      netSpec: { layers: [2, 3] },
    })
    pool.setDataset(dataset)
    const ga = pool.ga
    const inputs = dataset[0]
    let d3: Float32Array
    let d2: Float32Array
    for (let i = 0; i < Round; i++) {
      ga.evolve()
      let { fitness, gene } = best(ga.options)
      d3 = encode(gene, inputs)
      d2 = decode(gene, d3)
      console.dir(
        {
          round: (i + 1).toLocaleString(),
          fitness: fitness / N,
          inputs,
          d3,
          d2,
        },
        { depth: 20 },
      )
    }
  }).timeout(1000 * 60 * 60)
})
