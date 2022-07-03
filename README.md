# ga-autoencoder

Library to build and run auto-encoder neural network powered by genetic algorithm or GPU.

[![npm Package Version](https://img.shields.io/npm/v/ga-autoencoder)](https://www.npmjs.com/package/ga-autoencoder)

Two modes are supported currently.

**brain.js based**: neural network with GPU support

**ga-island based**: compact neural network with mirrored weightings and biases (half amount of parameters to be trained)

Usage examples reference to [ga.spec.ts](src/ga.spec.ts) and [brain.spec.ts](src/brain.spec.ts)

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
