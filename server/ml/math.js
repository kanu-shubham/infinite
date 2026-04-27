"use strict";

// Minimal linear algebra for small (<= ~30 dim) feature vectors.
// All matrices are arrays-of-arrays (row-major). Vectors are 1D arrays.

function zeros(n, m) {
  if (m == null) return Array(n).fill(0);
  return Array.from({ length: n }, () => Array(m).fill(0));
}

function identity(n) {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

function clone(A) {
  return A.map((row) => (Array.isArray(row) ? row.slice() : row));
}

function transpose(A) {
  const n = A.length;
  const m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

function matMul(A, B) {
  const n = A.length;
  const k = A[0].length;
  const m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
    }
  }
  return C;
}

function matVec(A, v) {
  const n = A.length;
  const m = v.length;
  const out = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += A[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

function vecAdd(a, b) {
  return a.map((x, i) => x + b[i]);
}

function vecSub(a, b) {
  return a.map((x, i) => x - b[i]);
}

function vecScale(a, s) {
  return a.map((x) => x * s);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function outer(a, b) {
  const n = a.length;
  const m = b.length;
  const M = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) M[i][j] = a[i] * b[j];
  return M;
}

function matAdd(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

function matScale(A, s) {
  return A.map((row) => row.map((v) => v * s));
}

// Gauss-Jordan inverse with partial pivoting. Throws on singular matrix.
function inverse(A) {
  const n = A.length;
  const M = clone(A).map((row, i) => row.concat(identity(n)[i]));
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
    }
    if (Math.abs(M[pivot][i]) < 1e-12) {
      throw new Error("Matrix is singular or nearly singular");
    }
    if (pivot !== i) {
      const tmp = M[i];
      M[i] = M[pivot];
      M[pivot] = tmp;
    }
    const pv = M[i][i];
    for (let j = 0; j < 2 * n; j++) M[i][j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = M[r][i];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= factor * M[i][j];
    }
  }
  return M.map((row) => row.slice(n));
}

// Cholesky: A = L L^T for symmetric positive-definite A. Returns lower-tri L.
function cholesky(A) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) {
        const v = A[i][i] - s;
        if (v <= 0) throw new Error("Matrix not positive definite");
        L[i][j] = Math.sqrt(v);
      } else {
        L[i][j] = (A[i][j] - s) / L[j][j];
      }
    }
  }
  return L;
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

module.exports = {
  zeros,
  identity,
  clone,
  transpose,
  matMul,
  matVec,
  matAdd,
  matScale,
  vecAdd,
  vecSub,
  vecScale,
  dot,
  outer,
  inverse,
  cholesky,
  sigmoid,
};
