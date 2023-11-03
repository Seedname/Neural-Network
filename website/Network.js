function sigmoid(n) {
  return 1/(1+exp(-n));
}

class Network{
  constructor(layers) {
    this.layers = layers;
    
    this.weights = [];
    this.biases = [];
    
    for (let i = 1; i < layers.length; i++) {
      this.weights.push(new Matrix(layers[i], layers[i-1]));
      this.biases.push(new Matrix(layers[i], 1));
    }
    
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].randomize(0,1);
      this.biases[i].randomize(0,1);
    }
    
  }
  
  feed(data) {
    let input = Matrix.arrayToMatrix(data);
    // print(input)
    let activation = input;

    for (let l = 0; l < this.weights.length; l++) {
      let weight = this.weights[l];
      let bias = this.biases[l];

      let z = Matrix.add(Matrix.dot(weight, activation), bias);
      activation = Matrix.map(z, sigmoid);
      // print(activation);
    }
    
    return activation;
  }  
}