# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
import heapq

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []

    def push(self, item, priority, last, action):
        pair = (priority,item, last, action)
        heapq.heappush(self.heap,pair)

    def pop(self):
        (priority,item, last, action) = heapq.heappop(self.heap)
        return (priority,item, last, action)

    def isEmpty(self):
        return len(self.heap) == 0

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    return  [s,s,w,s,w,w,s,w]

def dfs_chen(problem, pos, visited, curPath):
    if problem.isGoalState(pos):
        return True
    for (nextState, action, _) in problem.getSuccessors(pos):  
        if nextState in visited:
            continue
        visited.append(nextState)
        if dfs_chen(problem, nextState, visited, curPath):
            curPath.insert(0, action)
            return True
    return False

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    visited =[problem.getStartState()]
    findPath = []
    dfs_chen(problem, problem.getStartState(), visited, findPath)
    print(findPath)
    return findPath

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    visited = [problem.getStartState()]
    d = {}
    q = util.Queue()
    q.push(problem.getStartState())
    goal = 0
    while not q.isEmpty():
        pos = q.pop()
        if problem.isGoalState(pos):
            goal = pos
            break
        for (nextState, action, cost) in problem.getSuccessors(pos):
            if nextState not in visited:
                d[nextState] = (pos, action)
                q.push(nextState)
                visited.append(nextState) 
    last = goal
    path = []
    while last != problem.getStartState():
        (parent, action) = d[last]
        path.insert(0, action)
        last = parent
    return path

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    visited = []
    d = {}
    q = PriorityQueue()
    q.push(problem.getStartState(), 0,0, 0)
    goal = 0
    while not q.isEmpty():
        (culCost,pos, last, lastAction)  = q.pop()
        visited.append(pos)
        d[pos] = (last, lastAction)
        if problem.isGoalState(pos):
            goal = pos
            break
        for (nextState, action, cost) in problem.getSuccessors(pos):
            if nextState not in visited:
                q.push(nextState, cost + culCost, pos, action)
    last = goal
    path = []
    while last != problem.getStartState():
        (parent, action) = d[last]
        path.insert(0, action)
        last = parent
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    defaultV = 100000000000000
    visited = []
    route = {}
    q = util.PriorityQueue()
    q.push(problem.getStartState(), 0)
    visited.append(problem.getStartState())
    gScore = {}
    gScore[problem.getStartState()] = 0
    goal = 0
    while not q.isEmpty():
    	pos = q.pop()
    	if problem.isGoalState(pos):
    		goal = pos
        	break
        visited.append(pos)
        for (nextState, action, cost) in problem.getSuccessors(pos):
            if nextState in visited:
            	continue
            tmpScore = gScore[pos] + cost
            if tmpScore >= gScore.get(nextState, defaultV):
            	continue
            route[nextState] = (pos, action)
            gScore[nextState] = tmpScore
            # if nextState in q.heap:
            # 	q.heap.remove(nextState)
            for (r, s) in q.heap:
            	if nextState == s:
            		q.heap.remove((r, s))
            		break
            q.push(nextState, tmpScore + heuristic(nextState, problem))
    last = goal
    path = []
    while last != problem.getStartState():
        (parent, action) = route[last]
        path.insert(0, action)
        last = parent
    return path



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
