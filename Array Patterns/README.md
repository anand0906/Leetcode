# this folder includes patterns like basic array, two pointer, sliding window, prefix sum

<h2>Count Subarrays With Score Less Than K</h2>
<a href="https://leetcode.com/problems/count-subarrays-with-score-less-than-k/description/">Problem : 2302</a>

```python
def solve(n,arr,k):
    ans=0
    for i in range(n):
        for j in range(i,n):
            sum=0
            for t in range(i,j+1):
                sum+=arr[t]
            length=j-i+1
            if(sum*length<k):
                ans+=1
    return ans

def solve(n,arr,k):
    ans=0
    for i in range(n):
        sum=0
        for j in range(i,n):
            sum+=arr[j]
            length=j-i+1
            if(sum*length<k):
                ans+=1
    return ans


def solve(n,arr,k):
    left,right=0,0
    sum=0
    ans=0
    while right<n:
        length=right-left+1
        sum+=arr[right]
        while sum*length>=k and left<=right:
            sum-=arr[left]
            left+=1
            length=right-left+1
        ans+=(right-left+1)
        right+=1
    return ans
        



class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n=len(nums)
        return solve(n,nums,k)
```

<h3>Comments</h3>

<ul>
	<li>Saw hints and solved</li>
</ul>