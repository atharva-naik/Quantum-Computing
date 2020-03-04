import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import random 

class qubit:
	def __init__(self, a = np.complex_(), b = np.complex_()):  #n is the number of qubits, i.e there are 2**n amplitudes for fully defining state of all vectors 
		self.val = np.zeros(2, dtype = np.complex_())
		self.val[0] = a
		self.val[1] = b
		self.n = 1 

		c = np.complex_()
		for i in range(0, 2**self.n):
			c = c + self.val[i]*np.conjugate(self.val[i])

		for i in range(2**self.n):
			self.val[i] = (self.val[i]/np.sqrt(c))

	def normalize(self):
		c = np.complex_()
		for i in range(0, 2**self.n):
			c = c + self.val[i]*np.conjugate(self.val[i])

		for i in range(2**self.n):
			self.val[i] = (self.val[i]/np.sqrt(c))
	
	def disp(self):
		print("The {} qubit system has state {} ".format(self.n, self.val))

	def measure(self, m): #m is the number of measurements to be done
		rv = np.arange(2**self.n)
		p = 1/(2**self.n)
		prob = np.full(2**self.n, p, dtype = float)
		
		for i in range(2**self.n):
			c = self.val[i]*np.conjugate(self.val[i])
			prob[i] = c

		#dist = sp.stats.rv_discrete(a = 0, b = (2**self.n)-1, values = (rv, prob), seed = np.random.RandomState)
		obs = random.choices(rv, weights = prob, k=m)
		dist = np.full(2**self.n, 0, dtype = int)
		for i in range(m):
			dist[obs[i]] += 1
	
		return dist
############################################################# This part plots the result of the measurements		
	def simplot(self, dist):
		l = []
		for i in range(2**self.n):
			str = bin(i)
			str = str[2:]
			if len(str) < self.n:
				for i in range(self.n - len(str)):
					str = '0' + str
			str = '|' + str + '>'
			l.append(str)

		plt.bar(l, dist, align = 'center', alpha = 0.5)
		plt.show()
##############################################################	
		#0th index is index of zero and so on

def concat(q1 = qubit(), q2 = qubit()):
	q3 = qubit()
	q3.n = q1.n+q2.n
	q3.val = np.kron(q1.val, q2.val) 

	return q3

class gate: #currently 
	def __init__(self, a = np.zeros(2, dtype = np.complex_()), b = np.zeros(2, dtype = np.complex_())):
		self.mat = np.eye(2, dtype = np.complex_)
		self.n = 1 #no. of qubits it acts upon

		c = abs(a[0]) + abs(a[1]) + abs(b[0]) + abs(b[1])

		if c > 0:
			self.mat[0][0] = a[0]
			self.mat[1][0] = a[1]
			self.mat[0][1] = b[0]
			self.mat[1][1] = b[1]
			
	def apply(self, q = qubit()):
		if q.n == self.n: 
			q.val = np.matmul(self.mat, q.val)

		else:
			print("Math error")

	def disp(self):
		print(self.mat)

class X(gate):
	def __init__(self):
		super(X, self).__init__()
		self.mat[0][1] = 1
		self.mat[1][0] = 1
		self.mat[0][0] = 0
		self.mat[1][1] = 0

class Y(gate):
	def __init__(self):
		super(Y, self).__init__()
		self.mat[0][0] = 0
		self.mat[1][1] = 0
		self.mat[1][0] = 1j
		self.mat[0][1] = -1j

class Z(gate):
	def __init__(self):
		super(Z, self).__init__()
		self.mat[0][0] = 1
		self.mat[1][1] = -1

class H(gate):
	def __init__(self):
		super(H, self).__init__()
		p = 1/np.sqrt(2)
		self.mat[0][0] = p 
		self.mat[0][1] = p
		self.mat[1][0] = p
		self.mat[1][1] = -p 


class Uf(gate):
	def __init__(self, k, f):
		super(Uf, self).__init__()	
		print(f)
		self.n = k
		self.mat = np.zeros((2**k, 2**k), dtype = np.complex_)

		for i in range(2**k):
			x = divmod(i,2)[0]
			y = divmod(i,2)[1]
			c = 2*x + XOR(f[x], y)			
				
			self.mat[c][i] = 1
			
def series(g1 = gate(), g2 = gate()): #enter gates in left to right order
	if g1.n == g2.n:
		g3 = gate()
		g3.n = g1.n
		g3.mat = np.matmul(g2.mat, g1.mat)

		return g3
	
	else:
		print("Math error")

def parallel(g1 = gate(), g2 = gate()):
	g3 = gate()
	g3.n = g1.n + g2.n
	g3.mat = np.kron(g1.mat, g2.mat)
	return g3

def XOR(x, y):
	if x == y:
		return 0
	else:
		return 1

class circuit: #lets you define the circuit just by mentioning series parallel combinations of gates
	pass

if __name__ == "__main__":

	print("_________________________Ignore the warnings_________________________")
	#dist = q.measure(1000)
	#print(dist)
	N = input("Enter value of n: ")
	N = int(N)

	f = np.full(2**N, 0)
	for i in range(2**N):
		f[i] = input("Enter the value of f({}) = ".format(i))

	print(f)
	U = Uf(N+1, f)
	U.disp()

	h = H()
	H = H()
	
	for i in range(N-1):
		H = parallel(H, h)

	H1 = parallel(H, h)

	I = gate()
	I.n = N
	I.mat = np.eye(2**(N-1), dtype = np.complex_)

	q0 = qubit(1, 0)
	q1 = qubit(0, 1)
	q0n = qubit(1, 0)
	for i in range(N-1):
		q0n = concat(q0n, q0)

	q = concat(q0n, q1)
	q.disp()
	H1.apply(q)
	U.apply(q)
	I = gate()
	HI = parallel(H, I)
	HI.apply(q)
	
	dist = q.measure(100)
	print(dist)

	is_const = 1 

	for i in range(2**(N+1)):
		if dist[i] >= 2 and i>1: 
			is_const = 0            #numerically set as 2, ideally should be zero


	if is_const == 0: 
		print("It is a balanced function")
	else:
		print("It is a constant function")
	
	flag = input("plot the simulated outcomes?")
	flag = bool(flag)
	if flag == 1:
		q.simplot(dist)
	# U.apply(q3)
	# q3.disp()
	# dist = q3.measure(100)
	# print(dist)
	




	#g.apply(q)
	#q.disp()
	#q.val = np.matmul(H, q.val)
	#q.disp()

# class qubit:
# 	def __init__(self, a = np.complex128(), b = np.complex128()):
# 		self.val = np.zeros(2, dtype = np.complex_)
# 		if a*(np.conjugate(a)) + b*(np.conjugate(b)) > 1:
# 			c = np.sqrt(a*(np.conjugate(a)) + b*(np.conjugate(b)))
# 			a = a/c
# 			b = b/c
# 		self.val[0] = a
# 		self.val[1] = b

# 	def normalize(self):
# 		a = self.val[0]
# 		b = self.val[1] 
# 		c = np.sqrt(a*(np.conjugate(a)) + b*(np.conjugate(b)))
# 		self.val[0] = a/c
# 		self.val[1] = b/c 

# 	def disp(self):
# 		print(self.val)
# 		#0th index is index of zero and so on 

# class C:
# 	def __init__(self, a=0, b=0):
# 		self.x = a
# 		self.y = b

# 	def disp(self):
# 		print("{}+{}i".format(self.x, self.y))

# def addC(c1 = C(), c2 = C()):
# 	c3 = C()
# 	c3.x = c1.x + c2.x
# 	c3.y = c1.y + c2.y

# 	return c3 

	# Z = np.array(((1, 0),(0, -1)), dtype = np.complex_)
	# X = np.array(((0, 1),(1, 0)), dtype = np.complex_)
	# Y = np.array(((0, -1j),(1j, 0)), dtype = np.complex_)
	# p = (1/np.sqrt(2))
	# H = np.array(((p, p),(p, -p)), dtype = np.complex_)

		# 	cdf = np.zeros((2**self.n), dtype = float)
		# for i in range (2**n)
		# 	c = self.val[i]*conjugate(self.val[i])
		# 	cdf[i] = cdf[i-1] + c



		# for i in range(m):

	#############Begin Deutsh algo####################
	
	# f = np.full(2, 0)
	# for i in range(2):
	# 	f[i] = input("Enter the value of f({}) = ".format(i))

	# print(f)
	# U = Uf(2, f)
	
	# h = H()
	# I = gate()

	# q1 = qubit(1, 0)
	# q2 = qubit(0, 1)
	# q3 = concat(q1, q2)
	# q3.disp()
	# h2 = parallel(h, h)
	# h2.apply(q3)
	# U.apply(q3)
	# h3 = parallel(h, I)
	# h3.apply(q3)
	# q3.disp()
	
	# dist = q3.measure(100)
	# print(dist)
    ###################################################
    #################Deutsch Hozsa algo################ 

	#q = qubit(1, 10)
	#q.disp()
	# g = X()
	# g1 = gate()
	# print(g.n)
	# g2 = parallel(g1, g)
	# g2.disp()






