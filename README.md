<h1>Continuous Subarrays</h1>
<p><strong>Problem :</strong><a href="https://leetcode.com/problems/continuous-subarrays/description/">Click Here</a></p>

```python
def solve(n,arr):
    count=0
    for i in range(n):
        for j in range(i,n):
            flag=False
            for k in range(i,j+1):
                for l in range(i,j+1):
                    if(abs(arr[k]-arr[l])>2):
                        flag=True
                        break
                if(flag):
                    break
            if(not flag):
                count+=1
    return count

def solve_2(n,arr):
    count=0
    for i in range(n):
        flag=False
        for j in range(i,n):
            for l in range(i,j):
                if(abs(arr[j]-arr[l])>2):
                    flag=True
                    break
            if(flag):
                break
            if(not flag):
                count+=1
    return count


def checkWindow(n,arr,size):
    mini,maxi=float('inf'),0
    count=0
    for i in range(n-size+1):
        mini,maxi=float('inf'),0
        for k in range(i,i+size):
            mini=min(arr[k],mini)
            maxi=max(arr[k],maxi)
        if(abs(mini-maxi)<=2):
            count+=1
    return count

from collections import deque
def checkWindowOptimized(n,arr,size):
    mini,maxi=deque(),deque()
    count=0
    for i in range(size):
        while mini and arr[i]<=arr[mini[-1]]:
            mini.pop()
        while maxi and arr[i]>=arr[maxi[-1]]:
            maxi.pop()
        mini.append(i)
        maxi.append(i)
    diff=abs(arr[mini[0]]-arr[maxi[0]])
    if(diff<=2):
        count+=1
    for i in range(size,n):
        while mini and mini[0]<=i-size:
            mini.popleft()
        while maxi and maxi[0]<=i-size:
            maxi.popleft()
        while mini and arr[i]<=arr[mini[-1]]:
            mini.pop()
        while maxi and arr[i]>=arr[maxi[-1]]:
            maxi.pop()
        mini.append(i)
        maxi.append(i)
        diff=abs(arr[mini[0]]-arr[maxi[0]])
        if(diff<=2):
            count+=1
    return count


def solve_3(n,arr):
    count=0
    for i in range(2,n+1):
        count+=checkWindowOptimized(n,arr,i)
    count+=n
    return count

def solve_4(n,arr):
    right,left=0,0
    freq={}
    count=0
    while right<n:
        freq[arr[right]]=(freq.get(arr[right],0)+1)
        while max(freq)-min(freq)>2 and left<=right:
            freq[arr[left]]-=1
            if(freq[arr[left]]==0):
                del freq[arr[left]]
            left+=1
        count+=(right-left+1)
        right+=1
    return count
```
<p><strong>Other Solutions :</strong><a href="https://leetcode.com/problems/continuous-subarrays/?envType=daily-question&envId=2024-12-14">Click Here</a></p>
