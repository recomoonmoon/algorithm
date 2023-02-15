import random
from collections import defaultdict
#Question 1
# 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
# 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
#
# 来源：力扣（LeetCode）
# 链接：https://leetcode.cn/problems/house-robber
# 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


#在本数组上操作，避免了内存开销
nums = [random.randint(1, 100) for i in range(10)]
def maxProfit(nums=[]):
    n = len(nums)
    if n == 0:
        return 0
    elif n == 1:
        return nums[0]
    else:
        nums[1] = max(nums[0], nums[1])
        i = 2
        while i < n:
            print(f"i = {i}")
            nums[i] = max(nums[i-1], nums[i-2] + nums[i])
            i+=1
        return nums[-1]
print(f"Q1: nums = {nums} \nmax profit = {maxProfit(nums)}")
#Question 2
# 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
# 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
# 你可以认为每种硬币的数量是无限的。
#
# 来源：力扣（LeetCode）
# 链接：https://leetcode.cn/problems/coin-change
# 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

def roundUpCoins(coins=[], amount=0):
    def dpFunction(op, ed):
        for i in range(op, ed):
            times = MAX
            if dp[i] == -1:
                for coin in coins:
                    if dp[i - coin] != -1 and i - coin >= 0:
                        times = min(dp[i - coin] + 1, times)
            if times != MAX:
                dp[i] = times
    MAX = 999999999
    dp = [-1 for i in range(amount + 1)]
    coins = [i for i in coins if i <= amount]
    #初始化coins数组，部分过大的硬币不需要
    if amount == 0:
        return 0
    elif len(coins) == 0:
        return -1
    #amount等于零，不需要硬币，coins为空会报错，此时无解（没有能找的硬币）
    for i in coins:
        dp[i] = 1
    minnum = min(coins)
    maxnum = max(coins)
    dpFunction(minnum, maxnum + 1)
    dpFunction(maxnum + 1, amount + 1)
    return dp[amount]
#Question 3
# 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
# 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
# 来源：力扣（LeetCode）
# 链接：https://leetcode.cn/problems/longest-increasing-subsequence
# 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
def lengthOfLIS(nums) -> int:
     dp = [1 for i in nums]
     assert len(nums) >= 1
     if len(nums) == 1:
         return 1
     else:
         for i in range(1, len(nums)):
             for j in range(i):
                 if nums[i] > nums[j]:
                     dp[i] = max(dp[j] + 1, dp[i])
         return max(dp)


nums = [1,3,6,7,9,4,10,5,6]
print(nums)
print(lengthOfLIS(nums))

#Question 4
# 在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。
# 火车票有 三种不同的销售方式 ：
# 一张 为期一天 的通行证售价为 costs[0] 美元；
# 一张 为期七天 的通行证售价为 costs[1] 美元；
# 一张 为期三十天 的通行证售价为 costs[2] 美元。
# 通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张 为期 7 天 的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。
print("Q4")
class Solution:
    def mincostTickets(self, days, costs) -> int:
        dp = [0 for i in range(366)]

        for i in range(len(days)):

            if days[i] < 7:
                dp[days[i]] = min(dp[days[i]-1] + costs[0], costs[1], costs[2])

            elif days[i] < 30:
                dp[days[i]] = min(dp[days[i]-1] + costs[0], dp[days[i]-7] + costs[1], costs[2])
                #print(f"dp[days[i]] + costs[0] {dp[days[i]] + costs[0]}\ndp[days[i - 7]] + costs[1] {dp[days[i]] + costs[0]}")
            else:
                dp[days[i]] = min(dp[days[i]-1] + costs[0], dp[days[i]-7] + costs[1], dp[days[i] - 30] + costs[2])
                #print(f"dp[days[i]] + costs[0] {dp[days[i]] + costs[0]}\ndp[days[i - 7]] + costs[1] {dp[days[i]] + costs[0]}\ndp[days[i] - 30] + costs[2] {dp[days[i] - 30] + costs[2]}")
            dp[days[i]:] = [dp[days[i]] for k in range(len(dp[days[i]:]))]
            #print(dp, "\n")
        return dp[-1]

    def numTrees(self, n: int) -> int:
        if n <= 1:
            return 1
        else:
            g = [0 for i in range(n + 1)]
            g[0] = 1
            g[1] = 1
            for i in range(2, n+1):
                for j in range(1, i+1):
                    g[i] += g[j-1]*g[i-j]
            return g[n]

    def trap1(self, height0) -> int:
        n = len(height0)
        def getWater(i, left):
            if i < n - 1:
                right = max(height0[i+1:])
                water = min(left, right) - height0[i]
                if water > 0:
                    return water
                else:
                    return 0
            else:
                return 0

        left = 0
        ans = 0

        for i in range(n):
            ans += getWater(i, left)
            left = max(left, height0[i])
        return ans
    def trap2(self, height) -> int:
        n = len(height)
        dp = [0 for i in range(n)]
        left = height[0]
        for i in range(0, n, 1):
            dp[i] = max(left - height[i], 0)
            left = max(left, height[i])
        right = height[n-1]
        for i in range(n-1, -1, -1):
            dp[i] = min(max(right - height[i], 0), dp[i])
            right = max(right, height[i])
        return sum(dp)

    def superEggDrop(self, k: int, n: int) -> int:
        def cell(n):
            if int(n/2) == n/2:
                return int(n/2)
            else:
                return int(n/2) + 1
        if n == 1:
            return 0
        elif n == 2:
            return 1
        elif k == 1:
            return n
        else:
            return self.superEggDrop(k-1, cell(n)-1) + 1
    def subarraySum(self, nums, k: int) -> int:
        sm = 0
        mp = defaultdict(int)
        mp[0] = 1
        res = 0
        for i in nums:
            sm += i
            print(f"sum: {sm}\nres: {res}\n")
            res += mp[sm - k]
            mp[sm] += 1
        return res

    def maximalSquare(self, matrix) -> int:
        n1 = len(matrix)
        n2 = len(matrix[0])
        dp = [[0]*n2 for i in range(n1)]

        for i in range(n2):
            if matrix[0][i] == "1":
                dp[0][i] = 1


        for i in range(n1):
            if matrix[i][0] == "1":
                dp[i][0] = 1
        if min(n1, n2) >= 2:
            for i in range(1, n1):
                for j in range(1, n2):

                    if matrix[i][j] == "1":
                        dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
                    else:
                        dp[i][j] = 0

        return max(max(i) for i in dp)**2

    def maxProfit1(self, k: int, prices) -> int:
        n = len(prices)
        preprice = prices[0]
        alt = preprice
        trades = []
        for i in range(1, n):
            if prices[i] >= alt:
                alt = prices[i]
            else:
                trades.append(alt - preprice)
                preprice = prices[i]
                alt = preprice
        trades.append(alt - preprice)
        print(trades)
        trades = [i for i in trades if i > 0]
        if len(trades) <= k:
            return sum(trades)
        else:
            trades.sort()
            trades.reverse()
            return sum(trades[:k])

    def maxProfit(self, k: int, prices) -> int:
        if not prices:
            return 0

        n = len(prices)
        k = min(k, n // 2)
        buy = [[0] * (k + 1) for _ in range(n)]
        sell = [[0] * (k + 1) for _ in range(n)]

        buy[0][0], sell[0][0] = -prices[0], 0
        for i in range(1, k + 1):
            buy[0][i] = sell[0][i] = float("-inf")

        for i in range(1, n):
            buy[i][0] = max(buy[i - 1][0], sell[i - 1][0] - prices[i])
            for j in range(1, k + 1):
                buy[i][j] = max(buy[i - 1][j], sell[i - 1][j] - prices[i])
                sell[i][j] = max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);

        return max(sell[n - 1])

    def maxAreaOfIsland(self, grid) -> int:
        pass

    def containsDuplicate(self, nums) -> bool:
        haxi = defaultdict(int)
        for i in nums:
            if haxi[i] == 0:
                haxi[i] = 1
            else:
                return False
        return True



# days = [1,4,6,7,8,20]
# costs = [2,7,15]
s = Solution()
print(s.maxProfit(2, [3,2,6,5,0,3]))


