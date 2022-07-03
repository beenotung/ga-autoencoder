import { expect } from 'chai'
import { createBrain, splitBrain } from './brain'

describe('brain.js version', () => {
  it('should equal to decode(encode(input))', () => {
    const brain = createBrain({ netSpec: { layers: [2, 3] }, gpu: true })
    const dataset = [
      [-1, -1],
      [-0.5, -0.5],
      [0, 0],
      [0.5, 0.5],
      [1, 1],
    ]
    brain.train({ dataset })
    const net = brain.net
    const { encoder, decoder } = splitBrain(net)
    for (let input of dataset) {
      let mid = encoder.run(input)
      let output = decoder.run(mid)

      expect(input).lengthOf(2)
      expect(mid).lengthOf(3)
      expect(output).lengthOf(2)

      let margin = 0.15
      expect(input[0]).closeTo(output[0], margin)
      expect(input[1]).closeTo(output[1], margin)

      margin = 0.1
      let e2eOutput = net.run(input)
      expect(e2eOutput).lengthOf(2)
      expect(e2eOutput[0]).closeTo(output[0], margin)
      expect(e2eOutput[1]).closeTo(output[1], margin)
    }
  })
})
