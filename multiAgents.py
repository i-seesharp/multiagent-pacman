# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import numpy as np

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def calculateDistance(self, a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = []
        foodDistances = []
        safeDistance = 2

        minGhostDistance = 0
        for i in range(len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition()
            dist = self.calculateDistance(ghostPos, newPos)
            ghostDistances.append(dist)
        if len(ghostDistances) > 0:
            minGhostDistance = min(ghostDistances)

        foodInNewPos = newFood.asList()
        foodInCurrentPos = (currentGameState.getFood()).asList()
        
        minFoodDistance = 0
        for i in range(len(foodInNewPos)):
            dist = self.calculateDistance(newPos, foodInNewPos[i])
            foodDistances.append(dist)
        if len(foodDistances) > 0:
            minFoodDistance = min(foodDistances)

        value = 0

        if action == Directions.STOP:
            value = value - 50 #DISINCENTIVIZE
            
        if len(foodInNewPos) < len(foodInCurrentPos):
            value = value + 100*(len(foodInCurrentPos) - len(foodInNewPos))
        else:
            value = value + (50.0/(minFoodDistance + 0.0001))

        value = value - (10.0/(minGhostDistance + 0.0001))
        #currGhostStates = currentGameState.getGhostStates()
        #currGhostDistances = []

        #for i in range(len(currGhostDistances)):
            


        allGhostsScared = True
        for i in range(len(newScaredTimes)):
            if newScaredTimes[i] == 0:
                allGhostsScared = False
                break
        if not allGhostsScared:
            if minGhostDistance < safeDistance:
                value = -np.inf


        return value

        
        
            

        
       
    
             
        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        maxMoves = self.depth * gameState.getNumAgents()
        return self.miniMax(0, gameState, maxMoves)[0]
    

    def miniMax(self, player, gameState, depthLeft):

        bestMove = None
        if self.isTerminal(gameState, depthLeft):
            return [bestMove, self.evaluationFunction(gameState)]

        if player == 0: #MAX
            value = -np.inf
        else:
            value = np.inf #MIN

        lstActions = gameState.getLegalActions(player)
        for i in range(len(lstActions)):
            newGameState = gameState.generateSuccessor(player, lstActions[i])
            newDepthLeft = depthLeft - 1
            totalAgents = newGameState.getNumAgents()
            if player == totalAgents - 1:
                newPlayer = 0
            else:
                newPlayer = player + 1

            newMove, newVal = self.miniMax(newPlayer, newGameState, newDepthLeft)


            if player == 0 and (newVal > value):
                bestMove = lstActions[i]
                value = newVal

            if player > 0 and (newVal < value):
                bestMove = lstActions[i]
                value = newVal

        return [bestMove, value]
  

    def isTerminal(self, gameState, depthLeft):
        flag = False

        
        flag = flag or gameState.isLose()
        flag = flag or gameState.isWin()
        flag = flag or (depthLeft == 0)

        return flag

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        curr_alpha = -np.inf
        curr_beta = np.inf
        maxMoves = self.depth * gameState.getNumAgents()
        return self.alphaBeta(0, gameState, curr_alpha, curr_beta, maxMoves)[0]
        
    def alphaBeta(self, player, gameState, alpha, beta, depthLeft):

        bestMove = None
        if self.isTerminal(gameState, depthLeft):
            return [bestMove, self.evaluationFunction(gameState)]

        if player == 0: #MAX
            value = -np.inf
        else: #MIN
            value = np.inf

        lstActions = gameState.getLegalActions(player)

        for i in range(len(lstActions)):
            newGameState = gameState.generateSuccessor(player, lstActions[i])
            newDepthLeft = depthLeft - 1
            totalAgents = newGameState.getNumAgents()
            if player == totalAgents - 1:
                newPlayer = 0
            else:
                newPlayer = player + 1

            newMove, newVal = self.alphaBeta(newPlayer, newGameState, alpha, beta, newDepthLeft)

            if player == 0: #MAX
                if value < newVal:
                    bestMove = lstActions[i]
                    value = newVal
                if value >= beta:
                    return [bestMove, value]
                alpha = max(value, alpha)
            else: #MIN 	
                if value > newVal:
                    bestMove = lstActions[i]
                    value = newVal
                if value <= alpha:
                    return [bestMove, value]
                beta = min(value, beta)

        return [bestMove, value]

    def isTerminal(self, gameState, depthLeft):
        flag = False

        
        flag = flag or gameState.isLose()
        flag = flag or gameState.isWin()
        flag = flag or (depthLeft == 0)

        return flag

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        return self.expectimax(0, gameState, 0)[0]
        
        

        
    def expectimax(self, player, gameState, n):
        
        bestMove = None
        if self.isTerminal(gameState, n):
            return [bestMove, self.evaluationFunction(gameState)]

        if player == 0: #MAX
            value = -np.inf
        else:   #CHANCE
            value = 0
        
         

        lstActions = gameState.getLegalActions(player)
        for i in range(len(lstActions)):
            newGameState = gameState.generateSuccessor(player, lstActions[i])
            totalPlayers = newGameState.getNumAgents()
            if player == totalPlayers - 1:
                newPlayer = 0
            else:
                newPlayer = player + 1
            newMove, newVal = self.expectimax(newPlayer, newGameState, n+1)

            if player == 0 and (newVal > value):
                bestMove = lstActions[i]
                value = newVal
                
            if player > 0:
                prob = 1.0/len(lstActions)
                value = value + newVal*prob

        return [bestMove, value]

        
    def isTerminal(self, gameState, n):
        flag = False

        
        flag = flag or gameState.isLose()
        flag = flag or gameState.isWin()
        flag = flag or (n == self.depth*gameState.getNumAgents())

        return flag
        
def manhattanDistance(a,b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION:
    We use 7 features of the state space and take their weighted sum to get an estimate of the state utility
    The weights are fine tuned using experimentation and hyperparameter tuning
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()

    if currentGameState.isLose():
        return -np.inf

    currentGhosts = currentGameState.getGhostStates()
    ghostScores = []
    for i in range(len(currentGhosts)):
        if currentGhosts[i].scaredTimer == 0:
            dist = manhattanDistance(currentGhosts[i].getPosition(), currentPos)
            ghostScores.append(-10.0/(dist+0.0001))
        else:
            dist = manhattanDistance(currentGhosts[i].getPosition(), currentPos)
            ghostScores.append(50.0/(dist+0.0001))
            
        

    foods = currentGameState.getFood()
    currentFoods = foods.asList()

    minFoodDistance = np.inf
    for i in range(len(currentFoods)):
        dist = manhattanDistance(currentFoods[i], currentPos)
        if dist < minFoodDistance:
            minFoodDistance = dist

    currentPellets = list(currentGameState.getCapsules())
    minPelletDistance = np.inf
    usePellets = True
    for i in range(len(currentPellets)):
        dist = manhattanDistance(currentPellets[i], currentPos)
        if dist < minPelletDistance:
            minPelletDistance = dist
    if len(currentPellets) == 0:
        minPelletDistance = 0

    meanFoodDistance = 0
    for i in range(len(currentFoods)):
        meanFoodDistance += manhattanDistance(currentFoods[i], currentPos)
    if len(currentFoods) != 0:
        meanFoodDistance = float(meanFoodDistance)/len(currentFoods)

    

    currentScore = currentGameState.getScore() 
    features = [len(currentFoods), currentScore, len(currentPellets), 1.0/(minFoodDistance+0.0001),sum(ghostScores),1.0/(minPelletDistance+0.0001), 1.0/(meanFoodDistance+0.0001)]
    weights = [-1.0, 1.0, -1.0, 10.0, 1.0, 2.0, 2.5]

    score = 0
    for i in range(len(weights)):
        
        if i==3 or i==len(weights)-1:
            if features[0] == 0:
                continue
        if i==5 or i==6:
            if features[2] == 0:
                continue
        score = score + weights[i]*features[i] + random.randint(0,10)

    if features[0] == 0:
        score = max(score, currentScore)

    return score

        
    
    
        

    

# Abbreviation
better = betterEvaluationFunction
