
def is_prime(n):
    """判断一个数是否为素数
    
    Args:
        n: 整数
        
    Returns:
        是否为素数
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def gcd(a, b):
    """计算最大公约数
    
    Args:
        a: 整数
        b: 整数
        
    Returns:
        a和b的最大公约数
    """
    while b:
        a, b = b, a % b
    return a
