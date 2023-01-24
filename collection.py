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

def fast_sort(nums, flag_left, flag_right):
    if len(nums) <= 1:
        return nums
    else:
        if flag_left >= flag_right:
            return
        i = flag_left
        j = flag_right
        base = nums[flag_left]
        while i < j:
            while nums[j] >= base and i < j:
                j-=1
            nums[i] = nums[j]
            while nums[i] < base and i < j:
                i+=1
            nums[j] = nums[i]
        nums[i] = base
        fast_sort(nums, flag_left, i-1)
        fast_sort(nums, i+1, flag_right)

def shellsort(nums=[], n=len(nums)):
    gap = n
    while(gap > 1):
        gap = int(gap/2)
        i = 0
        while(i < n - gap):
            end = i
            temp = nums[end + gap]
            while(end >= 0):
                if(temp < nums[end]):
                    nums[end + gap] = nums[end]
                    end -= gap
                else:
                    break
            nums[end + gap] = temp
            i+=1

def insertsort(nums, n=len(nums)):
    if n <= 1:
        return
    else:
        print(n)
        for i in range(1, n):
            print(f"{i}: ", nums)
            temp = i - 1
            numi = nums[i]
            while temp >= 0:
                if nums[temp] > numi:
                    nums[temp + 1] = nums[temp]
                    temp -= 1
                else:
                    break
            nums[temp + 1] = numi

def mergesort(nums, left, right):
    if left >= right:
        return
    else:
        mid = int((left+right)/2)
        mergesort(nums, left, mid)
        mergesort(nums, mid+1, right)
        pointer_left = left
        pointer_right = mid + 1
        ans = []
        while pointer_left <= mid or pointer_right <= right:
            if pointer_left > mid:
                ans.append(nums[pointer_right])
                pointer_right += 1
            elif pointer_right > right:
                ans.append(nums[pointer_left])
                pointer_left += 1
            else:
                if nums[pointer_left] >= nums[pointer_right]:
                    ans.append(nums[pointer_right])
                    pointer_right += 1
                else:
                    ans.append(nums[pointer_left])
                    pointer_left += 1
        for i in range(len(ans)):
            nums[left+i] = ans[i]


print(nums)
mergesort(nums, 0, len(nums)-1)
print(nums)