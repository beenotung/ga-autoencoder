import { expect } from 'chai'
import { best } from 'ga-island'
import { createGaPool } from './ga'
import { mirrorDecode, mirrorEncode, runMirrorNet } from './nn'

const { sqrt } = Math

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
    const dataset = [
      [-1, -1],
      [-0.5, -0.5],
      [0, 0],
      [0.5, 0.5],
      [1, 1],
    ]
    const pool = createGaPool({
      netSpec: { layers: [2, 3] },
    })
    pool.train({ dataset, iterations: 10_000 })
    const ga = pool.ga
    let net = best(ga.options).gene
    for (let input of dataset) {
      const mid = mirrorEncode(net, input)
      const output = mirrorDecode(net, mid)

      expect(input).lengthOf(2)
      expect(mid).lengthOf(3)
      expect(output).lengthOf(2)

      let margin = 0.4
      expect(input[0]).closeTo(output[0], margin)
      expect(input[1]).closeTo(output[1], margin)

      margin = 0.1
      let e2eOutput = runMirrorNet(net, input)
      expect(e2eOutput).lengthOf(2)
      expect(e2eOutput[0]).closeTo(output[0], margin)
      expect(e2eOutput[1]).closeTo(output[1], margin)
    }
  })
})
