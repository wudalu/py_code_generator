"""
测试文件
"""

def fibonacci(n):
    """计算斐波那契数列的第n个数
    
    Args:
        n: 位置索引，从0开始
        
    Returns:
        斐波那契数
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """计算阶乘
    
    Args:
        n: 正整数
        
    Returns:
        n的阶乘
    """
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
