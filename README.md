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

<br>
<br>

<h1>Maximum Average Pass Ratio</h1>
<p><strong>Problem :</strong> <a href="https://leetcode.com/problems/maximum-average-pass-ratio/description">Click Here</a></p>

```python
import heapq
def solve(n,classes,k):
    impact=[]
    for p,t in classes:
        currentRatio=p/t
        expectedRatio=(p+1)/(t+1)
        impact.append([-(expectedRatio-currentRatio),(p,t)])
    heapq.heapify(impact)
    while k>=1:
        greatImpact,(p,t)=heapq.heappop(impact)
        p+=1
        t+=1
        currentRatio=p/t
        expectedRatio=(p+1)/(t+1)
        heapq.heappush(impact,[-(expectedRatio-currentRatio),(p,t)])
        k-=1
    total=0
    for imp,(p,t) in impact:
        total+=(p/t)
    return round(total/n,5)
```
<p><strong>Solution :</strong><a href="https://leetcode.com/problems/maximum-average-pass-ratio/solutions/1108491/python-100-efficient-solution-easy-to-understand-with-comments-and-explanation">Click Here</a></p>

<br>
<br>

<h1>Add Two Numbers</h1>
<p><strong>Problem : </strong><a href="https://leetcode.com/problems/add-two-numbers/description/">Click Here</a></p>


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        carry=0
        head=None
        while l1 or l2:
            temp=0
            if(l1):
                temp+=l1.val
                l1=l1.next
            if(l2):
                temp+=l2.val
                l2=l2.next
            temp+=carry
            carry=temp//10
            node=ListNode(temp%10)
            if(head==None):
                head=node
            else:
                prev.next=node
            prev=node
        if(carry>0):
            prev.next=ListNode(carry)
        return head
```

<h1>Median of Two Sorted Arrays</h1>
<p><strong>Problem :</strong><a href="https://leetcode.com/problems/median-of-two-sorted-arrays/description/">Click Here</a></p>

```python
def solve(n,m,arr1,arr2):
    totalLength=n+m
    totalArray=arr1+arr2
    totalArray.sort()
    if(totalLength%2):
        return round(totalArray[totalLength//2],5)
    else:
        ind1=totalLength//2
        ind2=ind1-1
        return round((totalArray[ind1]+totalArray[ind2])/2,5)

def solve_2(n,m,arr1,arr2):
    ind1=(n+m)//2
    ind2=ind1-1
    v1,v2=0,0
    count=0
    left,right=0,0
    while left<n and right<m:
        if(arr1[left]<=arr2[right]):
            if(count==ind1):
                v1=arr1[left]
            if(count==ind2):
                v2=arr1[left]
            count+=1
            left+=1
        else:
            if(count==ind1):
                v1=arr2[right]
            if(count==ind2):
                v2=arr2[right]
            count+=1
            right+=1
    while left<n:
        if(count==ind1):
            v1=arr1[left]
        if(count==ind2):
            v2=arr1[left]
        count+=1
        left+=1
    while right<m:
        if(count==ind1):
            v1=arr2[right]
        if(count==ind2):
            v2=arr2[right]
        count+=1
        right+=1
    if((n+m)%2):
        return round(v1,5)
    else:
        return round((v1+v2)/2,5)
```

<br>
<br>

<h1>Final Array State After K Multiplication Operations I</h1>
<p><strong>Problem link :</strong><a href="https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-i/description/">Click Here</a></p>

```python
def solve(n,arr,k,multiplier):
    for i in range(k):
        mini=min(arr)
        ind=arr.index(mini)
        arr[ind]=arr[ind]*multiplier
    return arr

import heapq

def solve_2(n,arr,k,multiplier):
    heap=[[v,i] for i,v in enumerate(arr)]
    heapq.heapify(heap)
    for i in range(k):
        smallest,ind=heap[0]
        heapreplace(heap,[smallest*multiplier,ind])
    for val,ind in heap:
        arr[ind]=val
    return arr
```