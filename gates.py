import numpy as np

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
	pass
#0th index is index of zero and so on

def concat(q1 = qubit(), q2 = qubit()):
	q3 = qubit()
	q3.n = q1.n+q2.n
	q3.val = np.kron(q1.val, q2.val)

	return q3

class gate: #currently
	def __init__(self, a = np.complex_(), b = np.complex_(), c = np.complex_(), d = np.complex_()):
		self.mat = np.eye(2, dtype = np.complex_)
		self.n = 1 #no. of qubits it acts upon

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

class circuit: #lets you define the circuit just by mentioning series parallel combinations of gates
	pass

if __name__ == "__main__":


#f_0 = input("Enter the value of f(0) ")
#f_1 = input("Enter the value of f(1) ")

	q = qubit(1, 0)
	q.disp()
	g = X()
	g1 = gate()
	print(g.n)
	g2 = parallel(g1, g)
	g2.disp()
#g.apply(q)
#q.disp()
#q.val = np.matmul(H, q.val)
#q.disp()

# class qubit:
# def __init__(self, a = np.complex128(), b = np.complex128()):
# self.val = np.zeros(2, dtype = np.complex_)
# if a*(np.conjugate(a)) + b*(np.conjugate(b)) > 1:
# c = np.sqrt(a*(np.conjugate(a)) + b*(np.conjugate(b)))
# a = a/c
# b = b/c
# self.val[0] = a
# self.val[1] = b

# def normalize(self):
# a = self.val[0]
# b = self.val[1]
# c = np.sqrt(a*(np.conjugate(a)) + b*(np.conjugate(b)))
# self.val[0] = a/c
# self.val[1] = b/c

# def disp(self):
# print(self.val)
# #0th index is index of zero and so on

# class C:
# def __init__(self, a=0, b=0):
# self.x = a
# self.y = b

# def disp(self):
# print("{}+{}i".format(self.x, self.y))

# def addC(c1 = C(), c2 = C()):
# c3 = C()
# c3.x = c1.x + c2.x
# c3.y = c1.y + c2.y

# return c3

# Z = np.array(((1, 0),(0, -1)), dtype = np.complex_)
# X = np.array(((0, 1),(1, 0)), dtype = np.complex_)
# Y = np.array(((0, -1j),(1j, 0)), dtype = np.complex_)
# p = (1/np.sqrt(2))
# H = np.array(((p, p),(p, -p)), dtype = np.complex_)

# cdf = np.zeros((2**self.n), dtype = float)
# for i in range (2**n)
# c = self.val[i]*conjugate(self.val[i])
# cdf[i] = cdf[i-1] + c



# for i in range(m):