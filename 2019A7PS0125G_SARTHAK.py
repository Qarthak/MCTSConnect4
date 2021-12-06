from math import sqrt
import numpy as np
import copy
import random
import gzip

# class TreeNode:
# 	def __init__(self,gameState,parent,player,lastLine):
# 		if(parent!=None):
# 			self.hasParent=True
# 		else:
# 			self.hasParent=False
# 		self.parent=parent
# 		self.gameState=gameState
# 		self.encoding=EncodingOfGameState(gameState)
# 		self.player=player
# 		self.lastLine=lastLine
# 		self.listOfChildren={}
# 		self.isTerminalNode=isGameFinished(gameState)
# 		self.hasChildren=False
# 		self.score=0
# 		self.numOfPlayouts=0
		
		
# 	def addChild(self,line):
# 		if(isLineValid(line,self.gameState)):
# 			self.hasChildren=True
# 			tempGame=copy.deepcopy(self.gameState)
# 			addDisk(line,self.player,tempGame)
# 			self.listOfChildren[line]=TreeNode(tempGame,self,3-self.player,line)
			
# 	def getChild(self,line):
# 		return self.listOfChildren[line]
		
# 	def getParent(self):
# 		return self.parent
	
# 	def getGameState(self):
# 		return self.gameState
		
# 	def getEncoding(self):
# 		return self.encoding
		
# 	def getPlayer(self):
# 		return self.player
		
# 	def getScore(self):
# 		return self.score
		
# 	def getNumOfDraws(self):
# 		return self.numOfDraws
		
# 	def getNumOfPlayouts(self):
# 		return self.numOfPlayouts
		
# 	def getNumOfChildren(self):
# 		return len(self.listOfChildren)
		
# 	def getListOfChildren(self):
# 		return self.listOfChildren
		
# 	def getListOfChildrenLines(self):
# 		return list(self.listOfChildren.keys())
		
# 	def getListOfChildrenNodes(self):
# 		return list(self.listOfChildren.values())
		
# 	def setLineNumber(self,line):
# 		self.lastLine=line
	
# 	def setNumOfPlayouts(self,num):
# 		self.numOfPlayouts=num
		
# 	def setScore(self,num):
# 		self.score=num
		
# 	def getRandomChild(self):
# 		return self.listOfChildren[random.choice(list(self.listOfChildren.keys()))]
		
# class MCTS:
# 	def __init__(self,numberOfSimulations):
# 		self.C=0.5
# 		self.numberOfSimulations=numberOfSimulations
# 		self.line=None
# 		self.numOfPlayouts=0
# 		self.treeSoFar={}
		
# 	def search(self,root,game,player):
# 		if(self.treeSoFar.get(root.encoding)==None):
# 			self.treeSoFar[root.encoding]=root
# 		else:
# 			root=self.treeSoFar[root.encoding]
		
# 		self.numberOfLines=len(game[0])
# 		self.HeightOfLines=len(game)
# 		self.root=root
# 		self.game=copy.deepcopy(game)
# 		self.player=copy.deepcopy(player)

# 		for i in range(self.numberOfSimulations):
# 			leafNode = self.selection(copy.deepcopy(self.root))
# 			newNode = self.expansion(leafNode)
# 			result = self.simulation(newNode)
# 			self.root = self.backpropagation(leafNode,result)
# 			self.numOfPlayouts+=1
# 		#Return child with maximum plays
# 		try:
# 			# tempp=root.getListOfChildrenNodes()
# 			print(gap)
# 			ans=max(self.root.getListOfChildrenNodes(),key=lambda x:x.getNumOfPlayouts())
# 			print(ans.getScore(), ans.getNumOfPlayouts())
# 			return ans.lastLine
# 		except:
# 			print("No child")
# 			print(self.root.getGameState())
		
# 	def selection(self,node):
# 		if(node.hasChildren==False):
# 			print(node.getEncoding(),node.getNumOfPlayouts(),node.getScore())
# 			return node
# 		for child in node.getListOfChildrenNodes():
# 			# print("Well")
# 			if(child.getNumOfPlayouts()==0):
# 				if(np.random.random()<=0.3):
# 					child.setScore(0.5)
# 					child.setNumOfPlayouts(1)
# 				print("Exploration")
# 				return self.selection(child)
# 		# print("Welll")
# 		print("Going down")
# 		#Using UCB
# 		return self.selection( (max(node.getListOfChildrenNodes(),key=lambda x:
# 			(x.getScore()/x.getNumOfPlayouts() + self.C*math.sqrt(math.log(self.numOfPlayouts)/x.numOfPlayouts)))))
		
# 	def expansion(self,node):
# 		if(node.hasChildren==False):
# 			for i in range(self.numberOfLines):
# 				if(isLineValid(i,node.getGameState())):
# 					node.addChild(i)
# 					self.treeSoFar[node.encoding]=node
# 					# return node.getChild(i)
# 		else:
# 			print("WHY CHILD?")
		
# 		#In case node is terminal
# 		if(node.getNumOfChildren()==0):
# 			return node

# 		return node.getRandomChild()
		
# 	def simulation(self,node):
# 		simulatedGame=copy.deepcopy(node.getGameState())
# 		simulatedPlayer=copy.deepcopy(node.getPlayer())
# 		while(isGameFinished(simulatedGame) == False):
# 			line = np.random.randint(0,5)
# 			if(isLineValid(line,simulatedGame)):
# 				addDisk(line,simulatedPlayer, simulatedGame)
# 				simulatedPlayer=3-simulatedPlayer
# 		if(whoWon(simulatedGame)==node.getPlayer()):
# 			return 1
# 		elif(whoWon(simulatedGame)==0):
# 			return 0.5
# 		return -0.1
		
# 	def backpropagation(self,node,result):
# 		node.setNumOfPlayouts(node.getNumOfPlayouts()+1)
# 		node.setScore(node.getScore()+result)
# 		if(node.getParent()!=None):
# 			return self.backpropagation(node.getParent(),result)
# 		return node


# class Node:
# 	def __init__(self, player, move=None, parent=None, state=None):
# 		self.state = copy.deepcopy(state)
# 		self.parent = parent
# 		self.move = move
# 		self.untriedMoves = getMoves(state)
			
# 		self.childNodes = []
# 		self.wins = 0
# 		self.visits = 0
# 		self.player = player
		
# 	def selection(self):
# 		foo = lambda x: x.wins/x.visits + np.sqrt(2*np.log(self.visits)/x.visits)
# 		return sorted(self.childNodes, key=foo)[-1]
		
# 	def expand(self, move, state):
# 		# return child when move is taken
# 			# remove move from current node
# 		child = Node(move=move, parent=self, state=state)
# 		self.untriedMoves.remove(move)
# 		self.childNodes.append(child)
# 		return child

# 	def update(self, result):
# 		self.wins += result
# 		self.visits += 1
		
# def MCTS(currentState, itermax, player, currentNode=None):
# 	rootnode = Node(state=currentState,player=player)
# 	if currentNode is not None: rootnode = currentNode
	
	
# 	for i in range(itermax):
# 		node = rootnode
# 		state = copy.deepcopy(currentState)
		
# 		# selection
# 			# keep going down the tree based on best UCT values until terminal or unexpanded node
# 		while node.untriedMoves == [] and node.childNodes != []:
# 			node = node.selection()
# 			state.move(node.move)

# 		# expand
# 		if node.untriedMoves != []:
# 			m = random.choice(node.untriedMoves)
# 			addDisk(m, node.player, state)
# 			node = node.expand(m, state)
		
# 		# rollout
# 		while state.getMoves():
# 			state.move(random.choice(state.getMoves()))
			
# 		# backpropagate
# 		while node is not None:
# 			node.update(state.result(node.player))
# 			node = node.parent

		
# 	foo = lambda x: x.wins/x.visits
# 	sortedChildNodes = sorted(rootnode.childNodes, key=foo)[::-1]
# 	print("AI\'s computed winning percentages")
# 	for node in sortedChildNodes:
# 		print('Move: %s    Win Rate: %.2f%%' % (node.move+1, 100*node.wins/node.visits))
# 	print('Simulations performed: %s\n' % i)
# 	return rootnode, sortedChildNodes[0].move

class Node:
	def __init__(self, move=None, parent=None, state=None):
		self.state = state.Clone()
		self.parent = parent
		self.move = move
		self.untriedMoves = state.getMoves()
		self.childNodes = []
		self.wins = 0
		self.visits = 0
		self.player = state.player
		
	def selection(self):
		foo = lambda x: x.wins/x.visits + np.sqrt(2*np.log(self.visits)/x.visits)
		return sorted(self.childNodes, key=foo)[-1]
		
	def expand(self, move, state):
		child = Node(move=move, parent=self, state=state)
		self.untriedMoves.remove(move)
		self.childNodes.append(child)
		return child

	def update(self, result):
		self.wins += result
		self.visits += 1
		
def MCTS(currentState, itermax, currentNode=None):
	rootnode = Node(state=currentState)
	if currentNode is not None: rootnode = currentNode
	
	for i in range(itermax):
		node = rootnode
		state = currentState.Clone()
		
		while node.untriedMoves == [] and node.childNodes != []:
			node = node.selection()
			state.move(node.move)

		if node.untriedMoves != []:
			m = random.choice(node.untriedMoves)
			state.move(m)            
			node = node.expand(m, state)
		
		while state.getMoves():
			state.move(random.choice(state.getMoves()))
			
		while node is not None:
			node.update(state.result(node.player))
			node = node.parent
		
		
	foo = lambda x: x.wins/x.visits
	sortedChildNodes = sorted(rootnode.childNodes, key=foo)[::-1]
	print("AI\'s computed winning percentages")
	for node in sortedChildNodes:
		print('Move: %s    Win Rate: %.2f%%' % (node.move+1, 100*node.wins/node.visits))
	print('Simulations performed: %s\n' % i)
	return rootnode, sortedChildNodes[0].move

#Q-Learning
def EncodePiece(number,position):
	return (number)*(2**position)

def EncodingOfGameState(game):
	encoding=0
	for i in range(len(game[0])):
		for j in range(len(game)):
			encoding+=(EncodePiece(game[j][i],len(game)-1-j))*((2**(len(game)+1)-1)**i)
	return encoding

class QLearning():
	epsilon=0.01
	alpha=0.1
	gamma=0.9
	state=0
	R=0
	player=1
	
	def __init__(self,state,player,game):
		self.state=state
		self.player=player
		self.Q=np.zeros(((2**(len(game)+1)-1)**len(game[0]),len(game[0])))

	def learningMove(self,game):
		self.state=EncodingOfGameState(game)
		if np.random.random()<self.epsilon:
			a=np.random.randint(0,5)
		else:
			a=np.argmax(self.Q[self.state])
		while((not isLineValid(a,game)) and self.Q[self.state][a]>-9999):
			self.Q[self.state][a]=-100000
			self.Q[self.state][a]-=10
			a=np.argmax(self.Q[self.state])
		tempGame=copy.deepcopy(game)
		addDisk(a,self.player,tempGame)
		if(whoWon(tempGame)==self.player):
			self.R=1
		elif(whoWon(tempGame)==0):
			self.R=0
		else:
			self.R=-1

		nextState=EncodingOfGameState(tempGame)
		self.Q[self.state][a]=self.Q[self.state][a]+self.alpha*(self.R+self.gamma*np.max(self.Q[nextState])-self.Q[self.state][a])

		return a


#game Engine
def initializeGame(m,n):
	game=np.zeros((m,n),dtype=int)
	return game

def whoWon(game):
	for i in range(len(game)):
		for j in range(len(game[0])-3):
			if game[i][j]==game[i][j+1]==game[i][j+2]==game[i][j+3]!=0:
				return game[i][j]

	for i in range(len(game)-3):
		for j in range(len(game[0])):
			if game[i][j]==game[i+1][j]==game[i+2][j]==game[i+3][j]!=0:
				return game[i][j]
				
	for i in range(len(game)-3):
		for j in range(len(game[0])-3):
			if game[i][j]==game[i+1][j+1]==game[i+2][j+2]==game[i+3][j+3]!=0:
				return game[i][j]
				
	for i in range(len(game)-3):
		for j in range(len(game[0])-3):
			if game[i][j+3]==game[i+1][j+2]==game[i+2][j+1]==game[i+3][j]!=0:
				return game[i][j+3]
	
	return 0

def isGameFinished(game):
	if(whoWon(game)!=0):
		return True
	
	topProduct=1
	for i in game[0]:
		topProduct*=i
	if topProduct!=0:
		return True
	return False

def isLineValid(line,game):
	if line<0 or line>len(game[0])-1:
		return False
	if game[0][line]!=0:
		return False
	return True

def addDisk(line,player,game):
	for i in range(len(game)):
		n=len(game)-1
		if game[n-i][line]==0:
			game[n-i][line]=player
			break
		
def otherPlayer(player):
	return 3-player
	
def getMoves(state):
	ans=[]
	for i in range(len(state[0])):
		if(isLineValid(i,state)):
			ans.append(i)
	return ans


class Connect4:
	def __init__(self, ROW, COLUMN, LINE):
		self.bitboard = [0,0]
		self.dirs = [1, (ROW+1), (ROW+1)-1, (ROW+1)+1]
		self.heights = [(ROW+1)*i for i in range(COLUMN)]
		self.lowest_row = [0]*COLUMN
		self.board = np.zeros((ROW, COLUMN), dtype=int)
		self.top_row = [(x*(ROW+1))-1 for x in range(1, COLUMN+1)]
		self.ROW = ROW 
		self.COLUMN = COLUMN
		self.LINE = LINE
		self.player = 1 
		
	def Clone(self):
		clone = Connect4(self.ROW, self.COLUMN, self.LINE)
		clone.bitboard = copy.deepcopy(self.bitboard)
		clone.heights = copy.deepcopy(self.heights)
		clone.lowest_row = copy.deepcopy(self.lowest_row)
		clone.board = copy.deepcopy(self.board)
		clone.top_row = copy.deepcopy(self.top_row)
		clone.player = self.player
		return clone
		
	def move(self, col):
		m2 = 1 << self.heights[col] 
		self.heights[col] += 1 
		self.player ^= 1
		self.bitboard[self.player] ^= m2 
		self.board[self.lowest_row[col]][col] = self.player + 1 
		self.lowest_row[col] += 1 
	
	def result(self, player):
		if self.winner(player): return 1 
		elif self.winner(player^1): return 0 
		elif self.draw(): return 0.5 
	
	def isValidMove(self, col): 
		return self.heights[col] != self.top_row[col]
	
	def winner(self, color):
		for d in self.dirs:
			bb = self.bitboard[color]
			for i in range(1, self.LINE): 
				bb &= self.bitboard[color] >> (i*d)
			if (bb != 0): return True
		return False
	
	def PrintGrid(self):
		board = np.flip(self.board, axis=0)
		for row in board:
			for col in row:
				print(col,end="")
			print('\n',end="")
		print('\n',end="")

	
	def draw(self): 
		return not self.getMoves() and not self.winner(self.player) and not self.winner(self.player^1)
	
	def complete(self): 
		return self.winner(self.player) or self.winner(self.player^1) or not self.getMoves()
	
	
	def getMoves(self):
		if self.winner(self.player) or self.winner(self.player^1): return [] 
		
		listMoves = []
		for i in range(self.COLUMN):
			if self.lowest_row[i] < self.ROW: 
				listMoves.append(i)
		return listMoves

def get_input(board):
	while True: 
		try: 
			cin = int(input("Your move, [1-%s]. Ans: " % board.COLUMN))
			if cin < 1 or cin > board.COLUMN: raise ValueError
			if not board.isValidMove(cin-1): 
				print(cin)
				raise ValueError
			print()
			return cin-1
		except ValueError:
			print('Invalid spot. Try again')


def begin_game(board, order, maxIterBig, maxIterSmall):
	players = ['MCTS'+str(maxIterSmall), 'MCTS'+str(maxIterBig)]
	node = Node(state=board)
	while True: 
		if order == 0: 
			# col = get_input(board)
			node, col = MCTS(board, maxIterSmall, currentNode=node)
			print('First is MCTS(%s) played column %s\n' %(maxIterSmall,col+1))
		elif order == 1: 
			node, col = MCTS(board, maxIterBig, currentNode=node)
			print('MCTS(%s) played column %s\n' %(maxIterBig,col+1))
		board.move(col)
		board.PrintGrid()
		node = goto_childNode(node, board, col)
		order ^= 1
		if board.complete(): break
	if not board.draw(): print('%s won' % players[board.player])
	else: print('Draw')
	
	del(node)
	
def goto_childNode(node, board, move):
	for childnode in node.childNodes:
		if childnode.move == move:
			return childnode
	return Node(state=board)

isMCTS = 0

def main():
	# isMCTS = Input("Type 1 to see MCTS40 vs MCTS200")
	oROW, oCOLUMN = 6,7 # change size of board here
	oLINE = 4           # change number of in-a-row here
	order = 0           # 0 for Human to go first; 1 for AI to go first
	print("\n%s-IN-A-ROW (Size: %s by %s)\n" % (oLINE, oROW, oCOLUMN))
	c4 = Connect4(oROW, oCOLUMN, oLINE) # create Connect4 object
	c4.PrintGrid()

	max_itersBig = 200
	max_itersSmall = 40
	
	# begin_game(c4, order, max_itersBig, max_itersSmall)
	begin_game(c4, order, max_itersSmall, max_itersBig)
	
	# wins=0
	# draws=0
	# losses=0
	# for i in range(1):
	# 	temp = begin_game(c4, order, max_itersBig, max_itersSmall ) # start game
	# 	if temp == 1: wins += 1
	# 	elif temp == 0: losses += 1
	# 	else:
	# 		draws+=1
	# for i in range(1):
	# 	temp = begin_game(c4, 1-order, max_itersSmall, max_itersBig ) # start game
	# 	if temp == 0: wins += 1
	# 	elif temp == 1: losses += 1
	# 	else:
	# 		draws+=1





if __name__=='__main__':
	main()
	
#212222222212021222212212222201221