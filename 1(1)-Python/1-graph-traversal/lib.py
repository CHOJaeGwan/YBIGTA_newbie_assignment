from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        Args:
            n (int): 정점의 개수
        Returns:
            None
        """
        self.n = n
        # 구현하세요!
        self.adj_list: DefaultDict[int, List[int]] = defaultdict(list)
    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        Args:
            u (int): 정점 u
            v (int): 정점 v
        Returns:
            None
        """
        # 구현하세요!
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현
        Args:
            start (int): 시작 정점
        Returns:
            result (list[int]): DFS 탐색 결과 (정점 번호의 리스트)
        """
        # 구현하세요!
        visited: List[bool] = [False] * (self.n + 1)
        result = []
        
        def dfs_recursive(node: int) -> None:
            visited[node] = True
            result.append(node)
            for neighbor in sorted(self.adj_list[node]):
                if not visited[neighbor]:
                    dfs_recursive(neighbor)
        dfs_recursive(start)
        return result
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현
        Args:
            start (int): 시작 정점
        Returns:
            result (list[int]): BFS 탐색 결과 (정점 번호의 리스트)
        """
        # 구현하세요!
        visited: List[bool] = [False] * (self.n + 1)
        result = []
        
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(self.adj_list[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return result
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
