
import math
import numpy as np

from .a_star import AStar

class AStarGrid(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
            and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, cost_map, diagonal=True):
        self.map = cost_map
        self.width = cost_map.shape[1]
        self.height = cost_map.shape[0]

        self.diagonal = diagonal

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (y1, x1) = n1
        (y2, x2) = n2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def distance_between(self, n1, n2):
        """this method returns the average of both cells cost"""
        # cost = (self.map[n1] + self.map[n2]) / 2.0
        cost = self.map[n2]
        distance = np.linalg.norm(np.array(n1) - np.array(n2))
        return cost + distance

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 8 adjacent
        nodes that can be reached (=any adjacent coordinate that is not a wall)
        """

        lst = [(0, -1), (0, +1), (-1, 0), (+1, 0)]
        if self.diagonal:
            lst += [(+1, -1), (+1, +1), (-1, -1), (-1, +1)]

        y, x = node
        for i, j in lst:
            x1 = x + i
            y1 = y + j
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                    yield (y1, x1)



if __name__ == '__main__':
    cost_map = np.array([[0.1, 0.2, 0.3, 0.4],
                         [  3,   4,   1,   2],
                         [ 10,  20,  40,  30],
                         [400, 200, 100, 300]])
    a_star_grid = AStarGrid(cost_map)
    path = a_star_grid.astar((0, 0), (3, 3))
    print(path)