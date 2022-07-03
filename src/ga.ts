import { GaIsland } from 'ga-island'
import { Net, NetSpec, randomNet } from './nn'

const { random } = Math

export function create(spec: NetSpec) {
  const mutateAmount = 0.25
  const ga = new GaIsland<Net>({
    mutationRate: 0.2,
    randomIndividual: () => randomNet(spec),
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
    fitness:(a)=>{

		},
  })
  return ga
}
