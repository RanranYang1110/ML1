#-*- coding:utf-8 -*-
# @author: qianli
# @file: leetcode_test.py
# @time: 2019/08/05
def lss(stones):
    if len(stones) == 1:
        return stones[0]
    else:
        bb = sorted(stones, reverse=1)
        while len(bb) > 2:
            bb = sorted([bb[0]-bb[1]] + bb[2:])
        bb = sorted(bb, reverse=1)
        return bb[0]-bb[1]
aa = [2,7,4,1,8,1]
bb = lss(aa)


# for i in range(1, 31):
#     print(i)
#     b = x % 10
#     x = int(x / 10)
#     res.append(b)
#     if x < 10:
#         res.append(x)
#         break
x = -123
res = ''
if x > 0:
    li = 1
else:
    li = -1
x = x * li
while x>=10:
    b = x % 10
    x = int(x / 10)
    res += str(b)
res += str(x)
int(res) * li


def isPalindrome(x: int) -> bool:
    if x < 0:
        return False

    else:
        x1 = x
        res = ''
        while x >= 10:
            b = x % 10
            x = int(x / 10)
            res += str(b)
        res += str(x)
        res = int(res)
        if res == x:
            return True,res
        else:
            return False,res
mm = isPalindrome(121)
print(mm)
#%%
s = 1994
A = {1: 'I', 5: 'V', 10: 'X', 50: 'L',
     100: 'C', 500: 'D', 1000: 'M',
     4: 'IV', 9: 'IX', 40: 'IL', 90: 'IC',
     400: 'ID', 900: 'IM'}
XX = [1, 5, 10, 50, 100, 500, 1000]
XX = [1000, 500, 100, 50, 10, 5, 1]
B = {}
C = ''
for i in range(7):
    a = int(s / XX[i])
    s = s - a * XX[i]
    C += A[XX[i]] * a
#%%
A = {'I' : 1, 'V': 5, 'X' : 10, 'L':50, 'C': 100, 'D' : 500, 'M' : 1000,
     'IV' : 4, 'IX': 9 , 'XL' : 40, 'XC': 90, 'CD': 400, 'CM': 900}
aa = 'MCMXCIV'
m = 0
while aa != '':
    if aa[0:2] in A:
        m += A[aa[0:2]]
        aa = aa[2:]
        print('aa=',aa)
    else:
        m += A[aa[0]]
        aa = aa[1:]
# for i in range(len(aa)):
#     if aa[i:i+2] in A:
#         m += A[aa[i:i+2]]
#         aa = aa[i+2:]
#         print('aa=',aa)
#         if aa is '':
#             break
#     else:
#         m += A[aa[i]]
#     print(m)
#%%
s = '{[]}'
stack=[]
map={')':'(',']':'[','}':'{'}
muban = ["()", "[]", "{}"]
for i in range(len(s)):
    stack.append(s[i])
    if len(stack) >= 2 and stack[-2]+stack[-1] in muban:
        stack.pop()
        stack.pop()
len(stack) == 0
#%%
def searchInsert(nums, target):
    minValues = nums[0]
    maxValues = nums[-1]
    N = len(nums)
    if target <= minValues:
        return 0
    elif target > maxValues:
        return N
    else:
        minIndex = 0
        maxIndex = N
        while maxIndex - minIndex > 1:
            midIndex = int((maxIndex + minIndex)/2)
            print('Index', midIndex)
            if target > nums[midIndex]:
                minIndex = midIndex
            elif target < nums[minIndex]:
                maxIndex = midIndex
            elif target == nums[midIndex]:
                return midIndex
                break
        return minIndex + 1
nums = [1,4,6,7,8,9]
target = 6

print(searchInsert(nums, target))