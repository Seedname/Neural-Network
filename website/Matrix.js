function func(n) {
  return 1/(1-exp(-n));
}

class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
  
    this.data = [];
    for(let i = 0; i < this.rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < this.cols; j++) {
          this.data[i][j] = 0;
      }
    }
  }
  
  add(that) {
    if(typeof that != "number" && this.rows != that.rows && this.cols != that.cols) {
      print("Matrix rows and Columns not equal");
      return undefined;
    } 
    if(that instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            this.data[i][j] += that.data[i][j];
        }
       }
    } else if(that instanceof Number) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
            this.data[i][j] += that;
        }
      }
    }
  }
  
  static add(matrix1, matrix2) {
    if(matrix1.rows != matrix2.rows && matrix1.cols != matrix2.cols) {
      print("Matrix rows and Columns not equal");
      return undefined;
    } 
    
    let result = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
         result.data[i][j] = matrix1.data[i][j] + matrix2.data[i][j];
      }
    }
    
    return result;
  }
  
  randomize(min, max) {
    for(let i = 0; i < this.rows; i++) {
      for(let j = 0; j < this.cols; j++) {
        this.data[i][j] = random(min, max);
      }
    }
  }
  
  fill(value) {
    for(let i = 0; i < this.rows; i++) {
      for(let j = 0; j < this.cols; j++) {
        this.data[i][j] = value;
      }
    }
  }

  static map(that, func) {
    let result = new Matrix(that.rows, that.cols);
    for(let i = 0; i < that.rows; i++) {
      for(let j = 0; j < that.cols; j++) {
        result.data[i][j] = func(that.data[i][j]);
      }
    }
    return result;
  }
  
  static transpose(matrix) {
    let result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static dot(matrix1, matrix2) {
    if (matrix1.cols !== matrix2.rows) {
      console.error('Columns of matrix1 must match rows of matrix2.');
      return undefined;
    }
    let result = new Matrix(matrix1.rows, matrix2.cols);
    for (let i = 0; i < matrix1.rows; i++) {
      for (let j = 0; j < matrix2.cols; j++) {
        let sum = 0;
        for (let k = 0; k < matrix1.cols; k++) {
          sum += matrix1.data[i][k] * matrix2.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }


  static subtract(matrix1, matrix2) {
    if(matrix1.rows !== matrix2.rows && matrix1.cols !== matrix2.cols) {
        print("Matrix rows and Columns not equal");
        return undefined;
    } 
    let result = new Matrix(matrix1.rows, matrix1.cols);
    for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] = matrix1.data[i][j] - matrix2.data[i][j];
        }
    }
    return result;
  }
  
  static arrayToMatrix(arr) {
    let numRows = arr.length;
    let numCols = arr[0].length;
    let matrix = new Matrix(numRows, numCols);
    
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols; j++) {
        matrix.data[i][j] = arr[i][j];
      }
    }
    return matrix;
  }
  
  static matrixToArray() {
    let arr = [];
    for(let i = 0; i < this.rows; i++) {
      for(let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }
}




