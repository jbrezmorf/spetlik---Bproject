import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import random as rn
import time as t


class Shoot:
    def __init__(self, coord, v, n, extremes, time, F0, p):
        self.dt = 10 / n

        self.X = coord
        self.V = v
        self.n = n
        self.n2 = p*n
        self.F = []
        self.F0 = F0
        self.F.append(F0)
        self.p = p
        self.extremes = extremes
        self.time = time
        self.Y = []
        self.S = []
        self.a = 0
        self.d = 0
        self.NN = 10;                   #minimalni pocet MC simulaci, ktery staci k odhadu V0 a V1
        self.monteCarlo()



    #zakladni metoda zajistujici ovladani programu
    def monteCarlo(self):
        N = []
        result = []
        Y0 = self.sim(10)
        Y1 = self.level2(10)

        N = self.rate(Y0, Y1)
        result = self.count(N[0], N[1], Y0, Y1)

        self.variance(result)

        return result[0]

    # pocita rozptyl konecneho vysledku
    def variance(self, result):
        N0 = result[3]
        N1 = result[4]

        if(N0 ==0):
            DY = result[2] / N1

        elif(N1 == 0):
            DY = result[1] / N0

        else:
            DY = result[1] / N0 + result[2] / N1

        print("DY = ", DY)


    #realizuje druhou uroven metody MC, vraci vektor hodnot Y
    def level2(self, N1):
        Y_1 = []
        Y_0 = []
        for i in range(N1):
                F = self.wind(self.n2)
                w = self.cycle(F)
                #Y_n100.append(w)
                G = self.avergeF(F)
                z = self.cycle(G)
                Y_0.append(z)
                Y_1.append(w - z)

        #print("Y_n100 = ", Y_n100[i], " Y_n50 = ", Y_n50[i], " Y_n100-n50 = ", Y_n100_n50[i], "V1 =", V1[i])
        v1 = self.dispersion(Y_1)
        y1 = self.average(Y_1)
        # print("průměr f1 = ", self.average(Y_n100))
        # print("průměr f0 = ", self.average(Y_n50))
        # print("pruměr f1-f0 = ", self.average(Y_n100_n50))
        # print("rozptyl f0 = ", self.dispersion(Y_n50))
        # print("rozptyl f1 = ", self.dispersion(Y_n100))
        # print("rozptyl f1-f0 =", self.dispersion(Y_n100_n50))


        #print("v1", v1)

        return Y_1

    # jednourovnovou MC simulaci
    def sim(self, N):
        self.Y = []

        for i in range(N):
            F = self.wind(self.n)
            self.Y.append(self.cycle(F))

        self.a = self.average(self.Y)
        #print("PRUMER", self.a)
        #print("ROZPTYL", self.dispersion(self.Y))

        return self.Y

    # provadi v kazdem kroku MC metody vypocet souradnice y
    def cycle(self, F):
        x, y, t, n = 0, 0, 0, 0
        X = self.X
        V = self.V
        self.dt = 10 / len(F)

        for i in range(len(F)):
            X = X + self.dt * V
            V = V + self.dt * F[i]
            # self.F.append(self.f(X, V))
            # print("F",F[i])

            x = X[0]
            y = X[1]
            # print(x, y)

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

            t = self.dt * (i + 1)
        # if (t>= self.time):
        #    print("cas prekrocen")


        # print("x=" ,x)
        #print("y=",y)
        #print("cas",t)
        return y

    # vraci vektor zprumerovanych sil, podle pomeru, ktery je predan konstruktoru
    def avergeF(self, K):
        G = []
        for i in range((np.round((len(K) / self.p)).astype(int))):
            sum = 0
            for j in range(self.p):
                sum = sum + K[self.p * i + j]
            G.append(sum / self.p)
        return G

    # vraci prumer hodnot
    def average(self, k):
        return np.mean(k)

    # vraci rozptyl hodnot
    def dispersion(self, k):
        return np.var(k)

    # metoda vypocitava N0 a N1
    def rate(self,Y0,Y1):
        Y = 0
        x =0
        y =0


        vY0 = self.dispersion(Y0)
        vY1 = self.dispersion(Y1)

        N0 = np.ceil(vY0/self.n)
        N1 = np.ceil(vY1/self.n2)


        # cyklus slouzi k tomu aby N0 nebo N1 byly vetsi nez minimalni pocet kroků metody, ktery je potrebny pro zjisteni V0 a V1
        while (N0 <self.NN and N1 <self.NN):
            N0 = N0*2
            N1 = N1*2
        #print("N0 a N1", N0, N1)

        N0 = np.ceil(N0).astype(int)
        N1 = np.ceil(N1).astype(int)

        return [N0, N1]




    # vraci vysledek - vyslednou souradnici Y, rozptyly a pomer N0, N1
    def count(self, N0, N1, Y0, Y1):

        Y = []
        V0 = 0
        V1 = 0

        aY0 = self.average(Y0)
        aY1 = self.average(Y1)

        vY0 = self.dispersion(Y0)
        vY1 = self.dispersion(Y0)


        if(N0 <= self.NN and N1 <= self.NN):
            Y_pom0 = []
            Y_pom1 = []
            if (N0 >0 and N1>0):
                for i in range(N0):
                    Y_pom0.append(Y0[i])

                for i in range(N1):
                    Y_pom1.append(Y1[i])

                Y = self.average(Y_pom1) + self.average(Y_pom0)
                V0 = self.dispersion(Y_pom0)
                V1 = self.dispersion(Y_pom1)

            if(N0 >0 and N1 ==0):
                for i in range(N0):
                    Y_pom0.append(Y0[i])

                Y = + self.average(Y_pom0)
                V1 =0
                V0 = self.dispersion(Y_pom0)
            print("Y1", Y)

        if(N0 <= self.NN and N1 > self.NN):
            Y_pom1 = self.level2(N1 - self.NN)
            Y_pom0 = []

            if (N0 > 0):
                for i in range(N1):
                    Y_pom0.append(Y0[i])
                Y = self.average(Y_pom1 + Y1) + self.average(Y_pom0)
                V0 = self.dispersion(Y_pom0)

            else:
                Y = self.average(Y_pom1 + Y1)
                V0 = 0
                N0 = N0 + self.NN

            N1 = N1 + self.NN
            V1 = self.dispersion(Y_pom1 + Y1)
            print("Y2", Y)


        if(N0 >self.NN and N1 <= self.NN):
            Y_pom = self.sim(N0-self.NN)
            Y_pom1 = []
            if(N1 >0):
             for i in range(N1):
                Y_pom1.append(Y1[i])
             Y = self.average(Y_pom+Y0) + self.average(Y_pom1)

             V1= self.dispersion(Y_pom1)
            else:
             Y = self.average(Y_pom + Y0)

             V1 = 0
            N0 = N0 + self.NN
            V0 = self.dispersion(Y_pom + Y0)
            print("Y3", Y)

        if(N0 > self.NN and N1 > self.NN):

            Y_pom0 = self.sim(N0 - self.NN)
            Y_pom1 = self.level2(N1-self.NN)

            Y = self.average(Y_pom0 + Y0)+ self.average(Y_pom1 + Y1)

            V0 = self.dispersion(Y_pom0 + Y0)
            V1 = self.dispersion(Y_pom1 + Y1)
            N0 = N0 + self.NN
            N1 = N1 + self.NN
            print("Y4", Y)



        return np.array([Y, V0, V1, N0, N1])

    #vraci vektor vektoru sil
    def power(self, num):
        Z = [];
        for i in range(num):
            Z.append(self.f())
        return Z

    # uniform rozdeleni od -1 do 1
    def f(self):
        return self.F0 + np.linalg.norm(self.F0) * 10 * np.array([0, (2 * rn.random() - 1)])

    # normalni rozdeleni
    def f2(self, a, b):
        return self.F + np.linalg.norm(self.F) * 10 * np.array([0, np.random.randn()])

    # vraci vektor skalaru sil
    def wind(self, n):
        F_average = -1
        F_deviation = 0.5
        F = [F_average] * 2
        F[0] = F[0] + (rn.random() - 0.5) * 2 * F_deviation
        F[1] = F[1] + (rn.random() - 0.5) * 2 * F_deviation
        scale = F_deviation
        fraction = 0.2 # 0-1; 0 means perfect correlation
        new_F = []
        if (n >2):
         while len(F) < n:
            new_F = []
            scale *= fraction
            for i in range(len(F) - 1):
                shift = scale * 2 * (rn.random() - 0.5)
                new_F.append(F[i])
                new_F.append((F[i] + F[i + 1]) / 2 + shift)
            new_F.append(F[-1])
            F = new_F

         del new_F[n:]  # drop remaining items
        else:
          new_F = F
        return new_F


s = Shoot(np.array([0, 0]), np.array([10, 0]), 10,np.array([-100,200,-300,400]),10,np.array([0,-1]), 2)
s.monteCarlo()
Y = []
for i in range (10):
    Y.append(s.monteCarlo())


print("vysledny rozptyl", s.dispersion(Y))

