// @ts-ignore
import { Autoencoder } from './Autoencoder.js.js'

const callback = function (data) {
  console.log(data)
}

const autoencoder = new Autoencoder({
  hiddenLayerSize: 6,
  p: 0.05 /*Sparsity parameter.*/,
  beta: 0.3 /*Weight of the sparsity term.*/,
  learningRate: 0.9,
  threshold_value:
    undefined /* Optional threshold value for cost. Defaults to 1/(e^3). */,
  regularization_parameter: 0.001 /*Optional regularization parameter to prevent overfitting. Defaults to 0.01.*/,
  optimization_mode: {
    mode: 0,
  } /*Optional optimization mode for type of gradient descent. {mode:1, 'batch_size': <your size>} for mini-batch and {mode: 0} for batch. Defaults to batch gradient descent.*/,
  notify_count: 10 /*Optional value to execute the iteration_callback after every x number of iterations. Defaults to 100.*/,
  iteration_callback:
    callback /*Optional callback that can be used for getting cost and iteration value on every notify count. Defaults to empty function.*/,
  maximum_iterations: 500 /*Optional maximum iterations to be allowed before the optimization is complete. Defaults to 1000.*/,
})

let input = [0.3, 0.6, 0.9]

autoencoder
  .train_network([input], [input])
  .then(console.log('\nTraining done!\n'))

let res = autoencoder.predict_result([input])
console.log(res._data[0])
