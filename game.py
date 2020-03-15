import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

from random import randint
from math import sqrt
import numpy as np

class game():
	def __init__(self, vervose = False, Hsize = 20, Wsize = 20):
		self.vervose = vervose

		self.Hsize = Hsize
		self.Wsize = Wsize
		self.max_dist = sqrt(Hsize*Hsize + Wsize*Wsize)

		self.snake = [[4,4], [4,3], [4,2], [4,1]]
		self.prev_snake = [self.snake[:], self.snake[:], self.snake[:]]
		self.food = [4, 6]
		self.food = []
		while self.food == []:
			self.food = [randint(0, self.Hsize-1), randint(0, self.Wsize-1)]
			if self.food in self.snake: self.food = []

		self.prev_food = [self.food, self.food, self.food]

		self.prevMov = KEY_RIGHT

		self.score = 0

		self.state = 'playing'

	def get_score(self):
		return self.score

	def get_state(self):
		return self.state

	def get_matrix(self):
		mat = list()
		for i in range(self.Hsize):
			mat.append(list())
			for j in range(self.Wsize):
				a = 0
				if [i, j] == self.snake[0]:
					a = 1
				elif [i, j] in self.snake:
					a = 0.5
				elif [i, j] == self.food:
					a = -1
				mat[i].append(a)

		return mat

	def get_move_matrix(self):
		mat = list()
		for i in range(self.Hsize):
			mat.append(list())
			for j in range(self.Wsize):
				a = list()
				if [i, j] == self.snake[0]:
					a.append(0.8)
				elif [i, j] in self.snake:
					a.append(1)
				elif [i, j] == self.food:
					a.append(-1)
				else:
					a.append(0.1)

				for k in range(3):
					if [i, j] == self.prev_snake[k][0]:
						a.append(0.8)
					elif [i, j] in self.prev_snake[k]:
						a.append(1)
					elif [i, j] == self.prev_food[k]:
						a.append(-1)
					else:
						a.append(0.1)
				mat[i].append(a)

		return mat

	def get_step(self):
		state = list()
		d = np.linalg.norm(np.array(self.snake[0])-np.array(self.food))

		state.append(d/self.max_dist)

		state.append((self.snake[0][0]-self.food[0])/self.Hsize)
		state.append((self.snake[0][1]-self.food[1])/self.Wsize)

		state.append(self.snake[0][0]/self.Hsize)
		state.append((self.Hsize-1-self.snake[0][0])/self.Hsize)
		state.append(self.snake[0][1]/self.Wsize)
		state.append((self.Wsize-1-self.snake[0][1])/self.Wsize)

		dist=list()
		d=self.snake[3:]
		head=self.snake[0]
		for body in d:
			dist.append(sqrt((body[0]-head[0])*(body[0]-head[0]) +
					(body[1]-head[1])*(body[1]-head[1])))

		state.append((d[np.argmin(dist)][0]-head[0])/self.Hsize)
		state.append((d[np.argmin(dist)][1]-head[1])/self.Wsize)

		del d[np.argmin(dist)]
		del dist[np.argmin(dist)]

		for i in range(10):
			if d:
				state.append((d[np.argmin(dist)][0]-head[0])/self.Hsize)
				state.append((d[np.argmin(dist)][1]-head[1])/self.Wsize)
			else:
				state.append(state[7])
				state.append(state[8])

		return state


	def print_matrix(self):
		mat = self.get_matrix()

		istr = ''
		for i in range(self.Wsize+2):
			istr += '-'
		print(istr)
		for m in mat:
			st = '|'
			for l in m:
				if l == -1:
					st = st + '+'
				elif l == 1:
					st = st + 'O'
				elif l == 0.5:
					st = st + 'o'
				else:
					st = st + ' '
			print(st+'|')
		print(istr)

	def step(self, mov):
		reward = 0
		self.prev_snake.insert(0, self.snake[:])
		self.prev_food.insert(0, self.food[:])
		self.prev_snake.pop(-1)
		self.prev_food.pop(-1)

		newPos = [self.snake[0][0] + (mov == KEY_DOWN and 1) + (mov == KEY_UP and -1),
				  self.snake[0][1] + (mov == KEY_LEFT and -1) + (mov == KEY_RIGHT and 1)]

		if newPos[0] == -1 or newPos[0] == self.Hsize or newPos[1] == -1 or newPos[1] == self.Wsize:
			reward = -1
			self.state = 'lose'
		elif newPos in self.snake[1:]:
			if not((mov < 260 and self.prevMov < 260 or mov >= 260 and self.prevMov >= 260) and mov - self.prevMov != 0):
				reward = -1
				self.state = 'lose'
			else:
				reward = self.step(self.prevMov)[0]
		else:
			self.prevMov = mov
			self.snake.insert(0, newPos)

			if self.snake[0] == self.food:
				self.score += 1

				if self.score == self.Hsize*self.Wsize:
					reward = 10
					self.state = 'win'
				else:
					reward = 1
					
					self.food = []
					while self.food == []:
						self.food = [randint(0, self.Hsize-1), randint(0, self.Wsize-1)]
						if self.food in self.snake: self.food = []
			else:
				self.snake.pop()

				#d = np.linalg.norm(np.array(self.snake[0])-np.array(self.food))
				#d = d/self.max_dist

				#reward = (-d)/10
		
		return reward, self.state