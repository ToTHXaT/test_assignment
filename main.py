from dataclasses import dataclass
from random import Random
from time import time
from typing import TypeAlias
from queue import Queue
from itertools import pairwise

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


Tower: TypeAlias = tuple[int, int]


def with_chance(chance, value, alternative_value, r: Random):
    if r.random() < chance:
        return value

    return alternative_value


def is_in_range(t1: Tower, t2: Tower, _r: int) -> bool:
    # assuming that 'within' means that distance between towers is r
    # otherwise a proper grid is not forming
    _r += 1
    return abs(t1[0] - t2[0]) <= _r and abs(t1[1] - t2[1]) <= _r


FREE = 0
OBSTRUCTED = 1
TOWER = 2
COVERED = 3


@dataclass
class CityGrid:
    grid: list[list[bool]]
    towers: list[Tower]

    def __init__(self, n: int, m: int, coverage=0.3, *, random_state=None):
        self.n = n
        self.m = m
        self.coverage = coverage
        self.towers = []
        self.graph: dict[Tower, set[Tower]] = dict()
        if random_state is None:
            self.r = Random(time())
        else:
            self.r = Random(random_state)

        self.grid = [
            [with_chance(coverage, OBSTRUCTED, FREE, self.r) for _ in range(m)]
            for _ in range(n)
        ]

    def place_tower(self, x: int, y: int, range_: int):
        if self.grid[x][y] == OBSTRUCTED:
            raise Exception("Cannot place towers on obstructed blocks")

        for i in range(max(x - range_, 0), min(x + range_ + 1, self.n)):
            for j in range(max(y - range_, 0), min(y + range_ + 1, self.m)):
                if self.grid[i][j] == FREE:
                    self.grid[i][j] = COVERED

        self.grid[x][y] = TOWER

    def solve_tower_placement(self, tower_range: int):
        # solving the same task in 1d case for each row and column
        # placing towers on intersection of row and column solutions
        # repeat until done

        is_free = True

        while is_free:
            is_free = False
            for i in range(self.n):
                for j in range(self.m):
                    if self.grid[i][j] == FREE:
                        is_free = True

            x_tower_placements = set()
            for x in range(self.n):
                curr = 0
                while curr < self.m and self.grid[x][curr] != FREE:
                    curr += 1
                placing_cursor = min(curr + tower_range, self.m - 1)
                while curr < self.m:
                    if self.grid[x][placing_cursor] in (OBSTRUCTED, TOWER, COVERED):
                        placing_cursor -= 1
                    elif self.grid[x][placing_cursor] == FREE:
                        x_tower_placements.add((x, placing_cursor))
                        curr = placing_cursor + 1 + tower_range
                        while curr < self.m and self.grid[x][curr] != FREE:
                            curr += 1
                        placing_cursor = min(curr + tower_range, self.m - 1)
            y_tower_placements = set()
            for y in range(self.m):
                curr = 0
                while curr < self.n and self.grid[curr][y] != FREE:
                    curr += 1
                placing_cursor = min(curr + tower_range, self.n - 1)
                while curr < self.n:
                    if self.grid[placing_cursor][y] in (OBSTRUCTED, TOWER, COVERED):
                        placing_cursor -= 1
                    elif self.grid[placing_cursor][y] == FREE:
                        y_tower_placements.add((placing_cursor, y))
                        curr = placing_cursor + 1 + tower_range
                        while curr < self.n and self.grid[curr][y] != FREE:
                            curr += 1
                        placing_cursor = min(curr + tower_range, self.n - 1)

            tower_placements = x_tower_placements & y_tower_placements

            for x, y in tower_placements:
                self.place_tower(x, y, tower_range)
                self.towers.append((x, y))

    def make_graph(self, _r: int):
        graph = dict()
        for t in self.towers:
            graph[t] = set()

        for tower in self.towers:
            towers_in_range = filter((lambda x: is_in_range(x, tower, _r)), self.towers)
            for t in towers_in_range:
                if t != tower:
                    graph[tower].add(t)

        self.graph = graph

    def find_most_reliable_path(self, t1: Tower, t2: Tower) -> list[Tower]:
        """Finds and returns the most reliable path. Returns empty list if no path is present."""

        # using width first approach we guarantee that the path will be the shortest

        path = [t1]

        curr_tower = t1
        visited = set()
        q = Queue()

        try:
            all_adjacent = self.graph[t1]
        except KeyError:
            print("No such tower")
            return []

        for adjacent in all_adjacent:
            q.put((adjacent, [*path, adjacent]))

        while not q.empty():
            curr, _p = q.get()

            if curr in visited:
                continue

            if curr == t2:
                return _p

            visited.add(curr)

            all_adjacent = self.graph[curr]
            for adjacent in all_adjacent:
                q.put((adjacent, [*_p, adjacent]))

        return []

    def visualize(self, path=None, block=True):
        rows, cols = self.n, self.m

        row_labels = tuple(range(rows))
        col_labels = tuple(range(cols))

        colors = ListedColormap(["w", "k", "b", "g", "red", "pink", "yellow"])
        plt.matshow(np.array(self.grid).T, cmap=colors, norm=Normalize(vmin=0, vmax=6))

        for t1, t2l in self.graph.items():
            for t2 in t2l:
                plt.plot((t1[0], t2[0]), (t1[1], t2[1]), color="red")

        if path is not None and len(path) == 0:
            print("No path")
        elif path:
            for t1, t2 in pairwise(path):
                plt.plot((t1[0], t2[0]), (t1[1], t2[1]), color="white", linewidth=5)

        plt.xticks(range(cols), col_labels)
        plt.yticks(range(rows), row_labels)
        plt.show(block=block)


def example():
    cg = CityGrid(12, 12, 0.3)
    tower_range = 2
    cg.solve_tower_placement(tower_range)
    cg.make_graph(tower_range)
    cg.visualize(block=False)

    x, y, x1, y1 = map(int, input("Coords: ").split())
    res = cg.find_most_reliable_path((x, y), (x1, y1))
    cg.visualize(path=res, block=True)


example()
