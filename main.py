import matplotlib
import numpy as np
import heapq

from numpy.random import  randint
import matplotlib.pyplot as plt


class Node:
    def __init__(self, key, v1, v2):
        self.key = key
        self.v1 = v1
        self.v2 = v2

    def __eq__(self, other):
        return np.sum(np.abs(self.key - other.key)) == 0

    def __ne__(self, other):
        return self.key != other.key

    def __lt__(self, other):
        return (self.v1, self.v2) < (other.v1, other.v2)

    def __le__(self, other):
        return (self.v1, self.v2) <= (other.v1, other.v2)

    def __gt__(self, other):
        return (self.v1, self.v2) > (other.v1, other.v2)

    def __ge__(self, other):
        return (self.v1, self.v2) >= (other.v1, other.v2)


class DStarLite:
    def __init__(self, r_map, sx, sy, gx, gy):
        self.start = np.array([sx, sy])
        self.goal = np.array([gx, gy])
        self.k_m = 0
        self.rhs = np.ones((r_map.shape[0], r_map.shape[1])) * np.inf
        self.g = self.rhs.copy()
        self.graph = r_map
        self.sensed_map = np.zeros((len(r_map), len(r_map[0])))
        self.rhs[self.goal[0], self.goal[1]] = 0
        self.queue = []
        node = Node(self.goal, *self.CalculateKey(self.goal))
        heapq.heappush(self.queue, node)

    def CalculateKey(self, node):
        key = [0, 0]
        key[0] = min(self.g[node[0], node[1]], self.rhs[node[0], node[1]]) + self.h_estimate(self.start, node) + self.k_m
        key[1] = min(self.g[node[0], node[1]], self.rhs[node[0], node[1]])
        return key

    def UpdateVertex(self, u):
        if np.sum(np.abs(u - self.goal)) != 0:
            s_list = self.succ(u)
            min_s = np.inf
            for s in s_list:
                if self.cost(u, s) + self.g[s[0], s[1]] < min_s:
                    min_s = self.cost(u, s) + self.g[s[0], s[1]]
            self.rhs[u[0], u[1]] = min_s
        if Node(u, 0, 0) in self.queue:
            self.queue.remove(Node(u, 0, 0))
            heapq.heapify(self.queue)
        if self.g[u[0], u[1]] != self.rhs[u[0], u[1]]:
            heapq.heappush(self.queue, Node(u, *self.CalculateKey(u)))

    def ComputeShortestPath(self):
        while len(self.queue) > 0 and \
                heapq.nsmallest(1, self.queue)[0] < Node(self.start, *self.CalculateKey(self.start)) or \
                self.rhs[self.start[0], self.start[1]] != self.g[self.start[0], self.start[1]]:

            k_old = heapq.nsmallest(1, self.queue)[0]
            u = heapq.heappop(self.queue).key
            if k_old < Node(u, *self.CalculateKey(u)):
                heapq.heappush(self.queue, Node(u, *self.CalculateKey(u)))

            # u = heapq.heappop(self.queue).key

            elif self.g[u[0], u[1]] > self.rhs[u[0], u[1]]:
                self.g[u[0], u[1]] = self.rhs[u[0], u[1]]
                s_list = self.succ(u)
                for s in s_list:
                    self.UpdateVertex(s)
            else:
                self.g[u[0], u[1]] = np.inf
                s_list = self.succ(u)
                s_list.append(u)
                for s in s_list:
                    self.UpdateVertex(s)

    # fetch successors and predessors
    def succ(self, u):
        s_list = [np.array([u[0] - 1, u[1] - 1]), np.array([u[0] - 1, u[1]]), np.array([u[0] - 1, u[1] + 1]),
                  np.array([u[0], u[1] - 1]), np.array([u[0], u[1] + 1]), np.array([u[0] + 1, u[1] - 1]),
                  np.array([u[0] + 1, u[1]]), np.array([u[0] + 1, u[1] + 1])]
        row = len(self.graph)
        col = len(self.graph[0])
        real_list = []
        for s in s_list:
            if 0 <= s[0] < row and 0 <= s[1] < col:
                real_list.append(s)
        return real_list

    #heuristic estimation
    def h_estimate(self, s1, s2):
        x_dist = s1[0] - s2[0]
        y_dist = s1[1] - s2[1]
        dist = np.sqrt(x_dist**2 + y_dist**2)
        return dist
    # def h_estimate(self, s1, s2):
    #     return np.linalg.norm(s1 - s2)

    # calculate cost between nodes
    def cost(self, u1, u2):
        if self.sensed_map[u1[0], u1[1]] == np.inf or self.sensed_map[u2[0], u2[1]] == np.inf:
            return np.inf
        else:
            return self.h_estimate(u1, u2)

    def sense(self, range_s):
        real_list = []
        row = len(self.graph)
        col = len(self.graph[0])
        for i in range(-range_s, range_s + 1):
            for j in range(-range_s, range_s + 1):
                if 0 <= self.start[0] + i < row and 0 <= self.start[1] + j < col:
                    if not (i == 0 and j == 0):
                        real_list.append(np.array([self.start[0] + i, self.start[1] + j]))
        return real_list


def ScanAndUpdate(node, last):
    s_list = node.sense(2)
    flag = True
    for s in s_list:
        if node.sensed_map[s[0], s[1]] != node.graph[s[0], s[1]]:
            flag = False
            # print('See a wall!')
            break
    if not flag:
        node.k_m += node.h_estimate(last, node.start)
        last = node.start.copy()
        for s in s_list:
            if node.sensed_map[s[0], s[1]] != node.graph[s[0], s[1]]:
                plt.plot(s[0], s[1], 'xr')
                node.sensed_map[s[0], s[1]] = node.graph[s[0], s[1]]
                node.UpdateVertex(s)
        for i in range(len(node.queue)):
            u = heapq.heappop(node.queue).key
            temp = Node(u, *node.CalculateKey(u))
            heapq.heappush(node.queue, temp)
        heapq.heapify(node.queue)
        node.ComputeShortestPath()
    return last


def maze(width, height, complexity=.05, density=.05):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * (shape[0] // 2 * shape[1] // 2))

    # Build actual maze
    z = np.zeros(shape, dtype=float)
    # Fill borders
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1]= 1
    # Make isles
    for i in range(density):
        x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
        z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[randint(0, len(neighbours) - 1)]
                if z[y_, x_] == 0:
                    z[y_, x_] = 1
                    z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return z


if __name__ == "__main__":
    # set start and goal point
    sx = 0
    sy = 0
    gx = 19
    gy = 19

    # set obstable positions
    graph = maze(width=20, height=20)
    ox, oy = [], []
    for i in range(0, len(graph)):
        for j in range(0, len(graph[i])):
            if graph[i][j] == 1:
                ox.append(i)
                oy.append(j)

    graph[graph == 1] = np.inf
    matplotlib.rc('figure', figsize=(5, 5))
    plt.grid(True)
    plt.plot([sx, gx], [sy, gy], 'r')
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")

    dstar = DStarLite(graph, sx, sy, gx, gy)

    last = dstar.start
    last = ScanAndUpdate(dstar, last)
    dstar.ComputeShortestPath()
    while np.sum(np.abs(dstar.start - dstar.goal)) != 0:
        s_list = dstar.succ(dstar.start)
        min_s = np.inf
        for s in s_list:
            # plt.plot(s[0], s[1], 'xy')
            if dstar.cost(dstar.start, s) + dstar.g[s[0], s[1]] < min_s:
                min_s = dstar.cost(dstar.start, s) + dstar.g[s[0], s[1]]
                temp = s
        dstar.start = temp.copy()
        # print(dstar.start[0], dstar.start[1])
        plt.plot(dstar.start[0], dstar.start[1], '.b')
        last = ScanAndUpdate(dstar, last)
        plt.pause(0.1)
    print("Goal Reached")

    plt.show()
