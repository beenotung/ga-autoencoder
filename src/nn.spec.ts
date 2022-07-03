import { expect } from 'chai'
import { tanh } from './math'
import {
  mirrorDecode,
  mirrorEncode,
  MirrorNet,
  randomMirrorNet,
  runMirrorNet,
} from './nn'

describe('randomNet()', () => {
  it('should return half network', () => {
    let net = randomMirrorNet({ layers: [2, 3] })
    expect(net.w).lengthOf(2 * 3, 'number of weight should be 2 x 3')
    expect(net.b).lengthOf(2 + 3, 'number of bias should be 2 + 3')
    expect(net.o).lengthOf(2, 'number of output layer should be 2')
    expect(net.o[0]).lengthOf(2, 'first output layer should have 2 outputs')
    expect(net.o[1]).lengthOf(3, 'first output layer should have 3 outputs')
  })
})

describe('runNet()', () => {
  let net: MirrorNet
  let inputs: number[]

  before(() => {
    /* [2,3] -> [2x3 | 3x2] */
    net = {
      w: new Float32Array([0.8, -0.8, 0.6, -0.6, 0.4, -0.4]),
      b: new Float32Array([0.8, 0.4, 0.0, -0.4, -0.8]),
      o: [new Float32Array([0, 0]), new Float32Array([0, 0, 0])],
    }

    inputs = [0.4, -0.8]
  })

  it('should be accurate', () => {
    let outputs = runMirrorNet(net, inputs)

    let h0 = tanh(inputs[0] * net.w[0] + inputs[1] * net.w[1] + net.b[2])
    let h1 = tanh(inputs[0] * net.w[2] + inputs[1] * net.w[3] + net.b[3])
    let h2 = tanh(inputs[0] * net.w[4] + inputs[1] * net.w[5] + net.b[4])
    let y1 = tanh(h2 * net.w[5] + h1 * net.w[3] + h0 * net.w[1] + net.b[1])
    let y0 = tanh(h2 * net.w[4] + h1 * net.w[2] + h0 * net.w[0] + net.b[0])
    let ys = [y0, y1]

    // console.dir({ net, inputs, outputs, ys }, { depth: 20 })

    let margin = 1e-7
    expect(outputs[0]).closeTo(ys[0], margin)
    expect(outputs[1]).closeTo(ys[1], margin)
  })

  it('should be equal to decode(encode(input))', () => {
    expect(runMirrorNet(net, inputs)).deep.equals(
      mirrorDecode(net, mirrorEncode(net, inputs))
    )
  })
})
