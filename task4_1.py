import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import random as rn
import time as t


class Shoot:
    def __init__(self, coord, v, n, extremes, time, F0, N):
        self.dt = 10 / n

        self.X = coord
        self.V = v
        self.n = n
        # np.ceil(10/dt).astype(int);
        # self.dt = dt;
        self.F = []
        self.F0 = F0
        self.F.append(F0)
        self.G = []
        self.N = N
        self.A = list()
        self.B = list()
        self.extremes = extremes
        self.time = time
        self.Y = []
        self.S = []
        self.a = 0
        self.d = 0

    def do(self):
        for i in range(10):
            self.sim()

            self.A.append(self.average(self.Y))
            self.B.append(self.dispersion(self.Y))
            # self.L = np.sort(self.Y)
            # self.M = []
            # self.M.append(self.L[np.ceil(len(self.L)/2).astype(int)])

        print(self.A, self.B)

        print("PRŮMĚR = ", self.average(self.A))
        # print("prumer medianu", self.average(self.M))
        print("ROPTYL PRUMERU = ", self.dispersion(self.A))
        print("průměr rozptylu", self.average(self.B))

        self.graph()
        return 0

    def graph(self):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(self.A)
        plt.show()
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(self.B)
        plt.show()
        return 0

    def sim(self):
        for i in range(self.N):
            F = self.wind()
            self.Y.append(self.cycle(F))

        # print("")
        # print("N =",self.N, "n=", self.time/self.dt)
        # for i in range (len(self.Y)):
        #      print("Y ´=", self.Y[i])
        self.a = self.average(self.Y)
        print("PRUMER", self.a)
        print("ROZPTYL", self.dispersion(self.Y))
        # self.histogram()
        return self.dispersion(self.Y)

    def cycle(self, F):
        x, y, t, n = 0, 0, 0, 0
        X = self.X
        V = self.V
        self.dt = 10 / len(F)

        for i in range(len(F)):
            X = X + self.dt * V
            V = V + self.dt * F[i]
            # self.F.append(self.f(X, V))
            #print("F",F[i])

            x = X[0]
            y = X[1]
            #print(x, y)

            if (x > self.extremes[1]):
                print("x je prilis velke")
                break
            if (x < self.extremes[0]):
                print("x je prilis male")
                break
            if (y > self.extremes[3]):
                print("y je prilis velke")
                break
            if (y < self.extremes[2]):
                print("y je prilis male")
                break

            n = n + 1
            # print("N =", n)
            t = self.dt * (i + 1)
        # if (t>= self.time):
        #    print("cas prekrocen")


        # print("x=" ,x)
        print("y=",y)
        print("cas",t)
        return y

    def avergeF(self, K):
        G = []
        for i in range((np.round((len(K) / 2)).astype(int))):
            G.append((K[2 * i] + K[2 * i + 1]) / 2)
        return G

    def average(self, k):
        return np.mean(k)

    def dispersion(self, k):


        return np.var(k)
    def getA(self):
        return self.a

    def histogram(self):
        p = np.ceil(np.sqrt(self.N)).astype(int)
        plt.figure(1)
        plt.hist(self.Y, p, normed=True)
        plt.xlim((min(self.Y), max(self.Y)))

        mean = self.average(self.Y)
        variance = self.dispersion(self.Y)
        sigma = np.sqrt(variance)
        x = np.linspace(min(self.Y), max(self.Y), 100)
        plt.plot(x, mlab.normpdf(x, mean, sigma))
        plt.show()

    def power(self):
        Z = [];
        for i in range(self.n):
            Z.append(self.f())

        return Z

    # uniform rozdeleni od -1 do 1
    def f(self):
        return self.F0 + np.linalg.norm(self.F0) * 10 * np.array([0, (2 * rn.random() - 1)])

    # normalni rozdeleni
    def f2(self, a, b):
        return self.F + np.linalg.norm(self.F) * 10 * np.array([0, np.random.randn()])

    def wind(self):
        F_average = -1
        F_deviation = 0.5
        F = [F_average] * 2
        F[0] = F[0] + (rn.random() - 0.5) * 2 * F_deviation
        F[1] = F[1] + (rn.random() - 0.5) * 2 * F_deviation
        scale = F_deviation
        fraction = 0.2  # 0-1; 0 means perfect correlation
        new_F = []
        if (self.n >2):
         while len(F) < self.n:
            new_F = []
            scale *= fraction
            for i in range(len(F) - 1):
                shift = scale * 2 * (rn.random() - 0.5)
                new_F.append(F[i])
                new_F.append((F[i] + F[i + 1]) / 2 + shift)
            new_F.append(F[-1])
            F = new_F

         del new_F[self.n:]  # drop remaining items
        else:
          new_F = F
        return new_F




#
# print("ALGO1")
# s = Shoot(np.array([0, 0]), np.array([10, 0]), 1000, np.array([-100, 200, -300, 400]), 10, np.array([0, -1]), 100)
# s.sim()

#print("ALGO2")
s = Shoot(np.array([0, 0]), np.array([10, 0]), 1, np.array([-100, 200, -300, 400]), 10, np.array([0, -1]), 10000)

V0 = []
A0 = []
a0 = 0
v0 = 0

for i in range(10):
    V0.append(s.sim())
    A0.append(s.getA())
a0 = s.average(A0)
v0 = s.average(V0)



# print("V0= ", v0)
# print("A0 =", a0)




