from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.
    Args:
        queue (deque[int]): 원형 큐
        k (int): 제거할 원소의 위치 (1부터 시작)
    Returns:
        removed_element (int): 제거된 원소
    """
    queue.rotate(-(k - 1)) 
    removed_element = queue.popleft()  
    return removed_element