const { exp } = Math

export function sigmoid(x: number): number {
  return 1 / (1 + exp(-x))
}

export function tanh(x: number): number {
  const ex = exp(x)
  const enx = exp(-x)
  return (ex - enx) / (ex + enx)
}
