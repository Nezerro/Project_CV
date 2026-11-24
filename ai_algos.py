"""Campus Watch: учебные алгоритмы ИИ без зависимостей от Qt.

Поиск пути (BFS, DFS, A*), антагонистический поиск (minimax и alpha-beta),
кластеризация (k-means++). Все функции чистые и тестируются отдельно.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

# Карта кампуса: # стена, . проход, G пост охраны, C камера
CAMPUS_MAP = [
    "######################",
    "#G....#.......#......#",
    "#.###.#.#####.#.####.#",
    "#.#...#.#...#...#....#",
    "#.#.###.#.#.#####.####",
    "#.#.....#.#.......#..#",
    "#.#######.#######.#..#",
    "#.....#...#.....#....#",
    "#####.#.###.###.#.####",
    "#.....#.....#...#....#",
    "#.#########.#.#####..#",
    "#...........C........#",
    "######################",
]
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def find_cell(grid, char) -> tuple:
    for y, row in enumerate(grid):
        x = row.find(char)
        if x >= 0:
            return x, y
    raise ValueError(f"на карте нет символа {char!r}")


def passable(grid, pos) -> bool:
    x, y = pos
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != "#"


def neighbors(grid, pos):
    x, y = pos
    for dx, dy in DIRS:
        nxt = (x + dx, y + dy)
        if passable(grid, nxt):
            yield nxt


def _restore(parent, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# --------------------------------------------------------------------------- #
# Поиск пути. Возвращают (путь или None, порядок посещения вершин)
# --------------------------------------------------------------------------- #
def bfs(grid, start, goal):
    """Поиск в ширину: очередь FIFO, кратчайший путь по числу шагов."""
    parent, order, queue = {start: None}, [], deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        if node == goal:
            return _restore(parent, goal), order
        for nxt in neighbors(grid, node):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return None, order


def dfs(grid, start, goal):
    """Поиск в глубину: стек LIFO. Путь находит, но не обязательно кратчайший."""
    parent, order, stack = {start: None}, [], [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        if node == goal:
            return _restore(parent, goal), order
        for nxt in neighbors(grid, node):
            if nxt not in seen:
                parent[nxt] = node
                stack.append(nxt)
    return None, order


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    """A*: f = g + h, h - манхэттенское расстояние (допустима и согласована на сетке)."""
    parent, best_g, order = {start: None}, {start: 0}, []
    heap, counter, closed = [(manhattan(start, goal), 0, start)], 0, set()
    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        order.append(node)
        if node == goal:
            return _restore(parent, goal), order
        for nxt in neighbors(grid, node):
            g = best_g[node] + 1
            if g < best_g.get(nxt, math.inf):
                best_g[nxt], parent[nxt] = g, node
                counter += 1
                heapq.heappush(heap, (g + manhattan(nxt, goal), counter, nxt))
    return None, order


ALGORITHMS = {"BFS": bfs, "DFS": dfs, "A*": astar}


# --------------------------------------------------------------------------- #
# Игра «патруль против нарушителя»: minimax и alpha-beta
# --------------------------------------------------------------------------- #
class PursuitGame:
    """Охранник (максимизатор, 2 клетки за ход) ловит нарушителя (минимизатор, 1 клетка).

    Оценка листа = -(кратчайшее расстояние по карте) : чем ближе охранник, тем лучше.
    Поимка (совпадение клеток) = большой плюс, причём чем раньше, тем лучше.
    """
    CATCH = 1000

    def __init__(self, grid):
        self.grid = grid
        self._dist_cache: dict = {}

    def dist(self, a, b) -> int:
        """Расстояние по карте (BFS от a, кэш по a)."""
        table = self._dist_cache.get(a)
        if table is None:
            table, queue = {a: 0}, deque([a])
            while queue:
                node = queue.popleft()
                for nxt in neighbors(self.grid, node):
                    if nxt not in table:
                        table[nxt] = table[node] + 1
                        queue.append(nxt)
            self._dist_cache[a] = table
        return table.get(b, 999)

    def moves(self, pos, fast=False):
        """Ходы: нарушитель - на соседнюю клетку или стоять; патруль (fast) - до 2 клеток за ход."""
        step1 = [pos, *neighbors(self.grid, pos)]
        if not fast:
            return step1
        cells = dict.fromkeys(step1)
        for cell in step1:
            for nxt in neighbors(self.grid, cell):
                cells[nxt] = None
        return list(cells)

    def best_move(self, guard, intruder, guard_turn=True, depth=4, alpha_beta=True):
        """Лучший ход текущего игрока + статистика (число вершин)."""
        stats = {"nodes": 0}
        score, move = self._run(guard, intruder, guard_turn, depth, alpha_beta, stats)
        return move, score, stats["nodes"]

    # alpha и beta передаются явными аргументами во внутренней функции - так нет скрытого состояния
    def _run(self, guard, intruder, guard_turn, depth, alpha_beta, stats):
        def rec(g, i, turn, d, alpha, beta):
            stats["nodes"] += 1
            if g == i:
                return self.CATCH + d, None
            if d == 0:
                return -self.dist(g, i), None
            best = None
            if turn:
                value = -math.inf
                for m in self.moves(g, fast=True):
                    s, _ = rec(m, i, False, d - 1, alpha, beta)
                    if s > value:
                        value, best = s, m
                    alpha = max(alpha, value)
                    if alpha_beta and alpha >= beta:
                        break
                return value, best
            value = math.inf
            for m in self.moves(i):
                s, _ = rec(g, m, True, d - 1, alpha, beta)
                if s < value:
                    value, best = s, m
                beta = min(beta, value)
                if alpha_beta and alpha >= beta:
                    break
            return value, best

        return rec(guard, intruder, guard_turn, depth, -math.inf, math.inf)


# --------------------------------------------------------------------------- #
# K-means по журналу событий
# --------------------------------------------------------------------------- #
def event_features(events):
    """Признаки события: время суток по кругу (sin, cos - 23:50 и 00:10 рядом) + тип события."""
    kinds = sorted({e.get("event", "") for e in events})
    rows, hours = [], []
    for e in events:
        try:
            moment = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        angle = 2 * math.pi * (moment.hour + moment.minute / 60) / 24
        onehot = [1.0 if e.get("event", "") == k else 0.0 for k in kinds]
        rows.append([math.sin(angle), math.cos(angle), *onehot])
        hours.append(moment.hour + moment.minute / 60)
    return np.array(rows, dtype=float), np.array(hours), kinds


def kmeans(data, k, seed=0, iters=100):
    """K-means с инициализацией k-means++. Возвращает (метки, центры, инерция)."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if not 1 <= k <= n:
        raise ValueError("k должно быть от 1 до числа точек")
    rng = np.random.default_rng(seed)
    centers = [data[rng.integers(n)]]
    while len(centers) < k:
        d2 = np.min([((data - c) ** 2).sum(axis=1) for c in centers], axis=0)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1 / n)
        centers.append(data[rng.choice(n, p=probs)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for step in range(iters):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if step and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = data[labels == j]
            if len(members):                       # пустой кластер оставляем на месте
                centers[j] = members.mean(axis=0)
    inertia = float(((data - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def circular_mean_hour(hours) -> float:
    angles = 2 * np.pi * np.asarray(hours) / 24
    hour = float((np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * np.pi)) * 24 / (2 * np.pi))
    return hour % 24.0 if hour < 23.9995 else 0.0


def cluster_events(events, k=3, seed=0):
    """Кластеризует события и описывает каждый кластер. None, если данных мало."""
    data, hours, kinds = event_features(events)
    if len(data) < max(k, 2):
        return None
    labels, _, inertia = kmeans(data, k, seed)
    clusters = []
    valid = [e for e in events if "time" in e]
    for j in range(k):
        idx = np.where(labels == j)[0]
        if not len(idx):
            continue
        names = [valid[i].get("event", "") for i in idx]
        top = max(set(names), key=names.count)
        mean_h = circular_mean_hour(hours[idx])
        clusters.append({"count": int(len(idx)), "hour": mean_h, "kind": top})
    clusters.sort(key=lambda c: -c["count"])
    return {"clusters": clusters, "inertia": inertia}


def elbow(events, max_k=6, seed=0):
    """Инерция для k=1..max_k: по «локтю» выбирают число кластеров."""
    data, _, _ = event_features(events)
    return [(k, kmeans(data, k, seed)[2]) for k in range(1, min(max_k, len(data)) + 1)]


def demo_events(n=90, seed=1):
    """Тестовые события для демонстрации: драки днём (перемены), падения утром и вечером."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kind = rng.choice(["Возможная драка", "Возможное падение", "Человек лежит"], p=[.4, .45, .15])
        centre = {"Возможная драка": 12.5, "Возможное падение": 8.5, "Человек лежит": 18.0}[kind]
        hour = float(np.clip(rng.normal(centre, 1.2), 0, 23.9))
        out.append({"time": f"2026-11-{int(rng.integers(1, 24)):02d} {int(hour):02d}:{int((hour % 1) * 60):02d}:00",
                    "event": str(kind), "confidence": float(rng.uniform(.5, .95))})
    return out
