import sys, os, csv, re
import numpy as np
import matplotlib.pyplot as plt

def open_csv(filename):
  X = []
  Y = []
  with open(filename) as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
      frow = [float(e) for e in row]
      X.append([frow[0], frow[1]])
      Y.append(frow[2])
  
  return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32))

def main():
  filename = sys.argv[1]
  X, Y = open_csv(filename)
  U, V = X[:, 0], X[:, 1]

  fig = plt.subplot(1, 2, 1)
  fig.scatter(U, V, s=0.5, c=Y, vmin=0, vmax=1)
  fig.axes.set_aspect('equal')

  fig = plt.subplot(1, 2, 2)
  data = list(zip(U, Y))
  W = np.array(sorted(data, key=lambda e: e[0]), dtype=np.float32)
  U, V = W[:, 0], W[:, 1]
  plt.plot(U, V)
  fig.set_ylim(-1, 2)

  plt.show()
  
if __name__ == "__main__":
  main()