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

<h1>Construct String With Repeat Limit</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/construct-string-with-repeat-limit/description/">Click Here</a></p>

```python
def solve(s,k):
    n=len(s)
    s=list(s)
    s.sort(reverse=True)
    i=0
    currentCount=0
    current=None
    while i<n:
        if(s[i]==current):
            currentCount+=1
        else:
            currentCount=1
            current=s[i]
        if(currentCount>k):
            j=i+1
            while j<n:
                if(s[j]!=current):
                    break
                j+=1
            else:
                return "".join(s[:i])
            s[i],s[j]=s[j],s[i]
            current=s[i]
            currentCount=1
        i+=1
    return "".join(s)

import heapq

def solve_2(s,k):
    count={-ord(i):0 for i in set(s)}
    for i in s:
        count[-ord(i)]+=1
    heap=list(count.items())
    heapq.heapify(heap)
    final=""
    current=None
    currentCount=0
    while heap:
        asci,cnt=heapq.heappop(heap)
        char=chr(abs(asci))
        if(char==current):
            currentCount+=1
        else:
            current=char
            currentCount=1
        if(currentCount>k):
            
            if(heap):
                asci2,cnt2=heapq.heappop(heap)
                final+=chr(abs(asci2))
                if(cnt2>1):
                    heapq.heappush(heap,(asci2,cnt2-1))
                current=chr(abs(asci2))
                currentCount+=1
                heapq.heappush(heap,(asci,cnt))
            else:
                return final
        else:
            final+=char
            if(cnt>1):
                heapq.heappush(heap,(asci,cnt-1))
    return final
```

<br>
<br>

<h1>Final Prices With a Special Discount in a Shop</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/description/">Click Here</a></p>

```python
def solve(n,arr):
    ans=[]
    for i in range(n):
        for j in range(i+1,n):
            if(arr[j]<=arr[i]):
                ans.append(arr[i]-arr[j])
                break
        else:
            ans.append(arr[i])
    return ans

def solve_2(n,arr):
    def next_smaller(n, arr):
        ans = []
        stack = []
        for i in range(n-1, -1, -1):
            while stack and  arr[i] < stack[-1] :
                stack.pop()
            if stack:
                ans.append(stack[-1])
            else:
                ans.append(-1)
            stack.append(arr[i])
        return ans[::-1]
    temp=next_smaller(n,arr)
    ans=[]
    for i in range(n):
        if(temp[i]!=-1):
            ans.append(arr[i]-temp[i])
        else:
            ans.append(arr[i])
    return ans
```

<br>
<br>

<h1>Max Chunks To Make Sorted</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/max-chunks-to-make-sorted/description/">Click Here</a></p>

```python
def solve(n,arr):
    expectedSum=0
    currentSum=0
    count=0
    for i in range(n):
        currentSum+=arr[i]
        expectedSum+=i
        if(currentSum==expectedSum):
            count+=1
    return count
```

<p><strong>Solution :</strong> <a href="https://leetcode.com/problems/max-chunks-to-make-sorted/description/comments/2275989">Click Here</a></p>

<br>
<br>

<h1>Reverse Odd Levels of Binary Tree</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/description/">Click Here</a></p>

```python
from collections import defaultdict
def solve(root):
    levels=defaultdict(list)
    queue=[(root,0)]
    levels[0].append(root)
    while queue:
        temp,level=queue.pop()
        if(temp.left):
            queue.append((temp.left,level+1))
            queue.append((temp.right,level+1))
            levels[level+1].extend([temp.left,temp.right])
    for i,v in levels.items():
        if(i&1):
            total=2**i
            temp=(2**i)//2
            for j in range(temp):
                v[j].val,v[total-j-1].val=v[total-j-1].val,v[j].val
    return root

def solve_2(root):
    queue=[root]
    odd=True
    level=0
    while queue:
        length=len(queue)
        for i in range(length):
            node=queue.pop(0)
            if(node.left):
                queue.append(node.left)
                queue.append(node.right)
        level+=1
        if(odd and queue):
            total=2**level
            temp=(2**level)//2
            for j in range(temp):
                queue[j].val,queue[total-j-1].val=queue[total-j-1].val,queue[j].val
        odd=not odd
    return root
```