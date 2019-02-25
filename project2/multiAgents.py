# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

    "*** YOUR CODE HERE ***"
    if successorGameState.isWin():
      return 1e10
    minDis = min([util.manhattanDistance(foodPos, newPos) for foodPos in newFood.asList()])
    disToGhost = [util.manhattanDistance(newPos, ghost) for ghost in successorGameState.getGhostPositions()]
    penalty = 0
    for d in disToGhost:
      if d <= 1:
        penalty += 1000000000
    return successorGameState.getScore() + 1/minDis + 1 /(1 + sum(disToGhost)) - penalty

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
  def maxLayer(self, gameState, depth):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(0):
      return self.evaluationFunction(gameState)
    action = [ self.minLayer(gameState.generateSuccessor(0, act), depth, 1) for act in gameState.getLegalActions(0) if act !=Directions.STOP ]
    return max(action)
  
  def minLayer(self, gameState, depth, agent):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agent):
      return self.evaluationFunction(gameState)
    if agent == gameState.getNumAgents() - 1:
      minCost = 100000000000000
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        minCost = min(minCost, self.maxLayer(nextState, depth + 1))
      return minCost
    else:
      minCost = 100000000000000
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        minCost = min(minCost, self.minLayer(nextState, depth, agent + 1))
      return minCost

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    action = [(self.minLayer(gameState.generateSuccessor(0, act), 0, 1), act) for act in gameState.getLegalActions(0) if act !=Directions.STOP ]
    return max(action)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def maxLayer(self, gameState, depth, alpha, beta):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(0):
      return self.evaluationFunction(gameState)
    v = -1e10
    for act in gameState.getLegalActions(0):
      if act == Directions.STOP:
        continue
      v = max(v, self.minLayer(gameState.generateSuccessor(0, act), depth, 1, alpha, beta))
      alpha = max(alpha, v)
      if alpha >= beta:
        return v
    return v
  
  def minLayer(self, gameState, depth, agent, alpha, beta):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agent):
      return self.evaluationFunction(gameState)
    if agent == gameState.getNumAgents() - 1:
      v = 1e10
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        v = min(v, self.maxLayer(nextState, depth + 1, alpha, beta))
        beta = min(beta, v)
        if alpha >= beta:
          return v
      return v
    else:
      v = 1e10
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        v = min(v, self.minLayer(nextState, depth, agent + 1, alpha, beta))
        beta = min(beta, v)
        if alpha >= beta:
          return v
      return v

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    action = [(self.minLayer(gameState.generateSuccessor(0, act), 0, 1, -1e10, 1e10), act) for act in gameState.getLegalActions(0) if act !=Directions.STOP ]
    return max(action)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def maxLayer(self, gameState, depth):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(0):
      return self.evaluationFunction(gameState)
    action = [ self.minLayer(gameState.generateSuccessor(0, act), depth, 1) for act in gameState.getLegalActions(0) if act !=Directions.STOP ]
    return max(action)
  
  def minLayer(self, gameState, depth, agent):
    if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agent):
      return self.evaluationFunction(gameState)
    if agent == gameState.getNumAgents() - 1:
      cost = 0
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        cost = cost + self.maxLayer(nextState, depth + 1)
      return cost / len(gameState.getLegalActions(agent))
    else:
      cost = 0
      for act in gameState.getLegalActions(agent):
        if act == Directions.STOP:
          continue
        nextState = gameState.generateSuccessor(agent, act)
        cost = cost + self.minLayer(nextState, depth, agent + 1)
      return cost / len(gameState.getLegalActions(agent))

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    action = [(self.minLayer(gameState.generateSuccessor(0, act), 0, 1), act) for act in gameState.getLegalActions(0) if act !=Directions.STOP ]
    return max(action)[1]


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  # if len(currentGameState.getFood().asList()) == 0:
  #   return 1e10
  
  # pos = currentGameState.getPacmanPosition()
  # score = max([0] + [1.0 / util.manhattanDistance(pos, foodPos) for foodPos in currentGameState.getFood().asList()])

  # return currentGameState.getScore() + score
  foodList = currentGameState.getFood().asList()
  pos = list(currentGameState.getPacmanPosition())
  ghostScore = 0
  for ghost in currentGameState.getGhostStates():
    dist = manhattanDistance(ghost.getPosition(),pos)
    if ghost.scaredTimer > 0:
      ghostScore += dist
    else:
      ghostScore -= dist
  disToFood = sum([util.manhattanDistance(food, pos) for food in foodList])
  disToCapsule = sum([util.manhattanDistance(cap, pos) for cap in currentGameState.getCapsules()])
  return currentGameState.getScore() - len(foodList) - len(currentGameState.getCapsules()) - disToFood - disToCapsule + ghostScore

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

