#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_searchInsert.py
# @time: 2019/08/07
def searchInsert(nums, target):
    minValues = nums[0]
    maxValues = nums[-1]
    N = len(nums)
    if target <= minValues:
        return 0
    elif target > maxValues:
        return N
    elif target in nums:
        return nums.index(target)
    else:
        minIndex = 0
        maxIndex = N
        while maxIndex - minIndex > 1:
            midIndex = int((maxIndex + minIndex)/2)
            print('Index', midIndex)
            if target > nums[midIndex]:
                minIndex = midIndex
            elif target < nums[midIndex]:
                maxIndex = midIndex
        return minIndex + 1
nums = [1,3,4,6,7,8,9]
target = 2

print(searchInsert(nums, target))
#%%
def mySqrt(x):
    if x < 1:
        return 0
    else:
        n = x/2
        for i in range(10):
            n = (n + x / n) / 2
        return int(n)
print(mySqrt(100000))
#%%
def twoSum(numbers, target):
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    dict_a = {}
    for i in range(len(numbers)):
        x = target - numbers[i]
        if numbers[i] not in dict_a:
            dict_a[numbers[i]] = i
        else:
            return [dict_a[numbers[i]], i]
numbers = [2, 7, 11, 15]
target = 9
print(twoSum(numbers, target))