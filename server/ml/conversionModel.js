"use strict";

const { sigmoid, dot } = require("./math");

// L2-regularized logistic regression for P(book | price, context).
// We fit with full-batch gradient descent — fine for the volume we get from
// in-memory feedback. For production this would move to SGD or LBFGS.
class LogisticRegression {
  constructor({ dim, l2 = 1.0, lr = 0.1, epochs = 200 }) {
    this.dim = dim;
    this.l2 = l2;
    this.lr = lr;
    this.epochs = epochs;
    this.weights = Array(dim).fill(0);
    this.bias = 0;
  }

  predictProba(x) {
    return sigmoid(this.bias + dot(this.weights, x));
  }

  fit(X, y) {
    const n = X.length;
    if (n === 0) return this;
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const grad = Array(this.dim).fill(0);
      let gBias = 0;
      for (let i = 0; i < n; i++) {
        const p = this.predictProba(X[i]);
        const err = p - y[i];
        gBias += err;
        for (let j = 0; j < this.dim; j++) grad[j] += err * X[i][j];
      }
      // average + L2 shrinkage
      for (let j = 0; j < this.dim; j++) {
        grad[j] = grad[j] / n + this.l2 * this.weights[j];
        this.weights[j] -= this.lr * grad[j];
      }
      this.bias -= this.lr * (gBias / n);
    }
    return this;
  }

  toJSON() {
    return { dim: this.dim, l2: this.l2, weights: this.weights, bias: this.bias };
  }

  static fromJSON(obj) {
    const m = new LogisticRegression({ dim: obj.dim, l2: obj.l2 });
    m.weights = obj.weights;
    m.bias = obj.bias;
    return m;
  }
}

module.exports = { LogisticRegression };
