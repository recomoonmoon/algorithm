import random
nums = [random.randint(1, 100) for i in range(10)]

def swap(nums, a, b):
    temp = nums[a]
    nums[a] = nums[b]
    nums[b] = temp

def heapinsert(nums, n=len(nums)):
    if n <= 1:
        return
    else:
        for i in range(1, n):
            current_index = i
            father_index = int(i-1/2)
            while current_index != 0 and not nums[father_index] >= nums[current_index]:
                swap(nums, father_index, current_index)
                current_index = father_index
                father_index = int(current_index-1/2)

def heapify(nums, target_index, size):
    current_index = target_index
    max_index = current_index
    max_num = nums[max_index]
    left = current_index*2 + 1
    right = current_index*2 + 2
    while left < size and (nums[current_index] < nums[left] or nums[current_index] < nums[right]):
        if right < size and nums[left] >= nums[right]:
            max_index = left
            max_num = nums[left]
        elif right < size and nums[right] > nums[left]:
            max_index = right
            max_num = nums[right]
        else:
            max_index = left
            max_num = nums[left]
        swap(nums, current_index, max_index)
        current_index = max_index
        left = current_index * 2 + 1
        right = current_index * 2 + 2

def heapSort(nums):
    #建立大根堆
    size = len(nums)
    print("建立堆前", nums)
    heapinsert(nums, size)
    print("建立堆后", nums)
    print(f"size: {size}")
    #每次把最大数放在末尾
    while size >= 1:
        swap(nums, 0, size-1)
        print(f"size: {size} 排序前 {nums}")
        heapify(nums=nums, target_index=0, size=size-1)
        print(f"size: {size} 排序后 {nums}")
        size -= 1

print(nums)
heapSort(nums)
print(nums)