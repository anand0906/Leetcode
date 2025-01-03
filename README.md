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

<h1>Maximum Number of K-Divisible Components</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/maximum-number-of-k-divisible-components/description">Click Here</a></p>

```python
def solve(n,edges,values,k):
    if(n<=1):
        return 1
    graph=defaultdict(set)
    for i,j in edges:
        graph[i].add(j)
        graph[j].add(i)
    q=deque()
    final=0
    for i,v in graph.items():
        if(len(v)==1):
            q.append(i)
    while q:
        for i in range(len(q)):
            node=q.popleft()
            parent=None
            if(graph[node]):
                parent=next(iter(graph[node]))
            if(parent!=None):
                graph[parent].remove(node)
            if(values[node]%k==0):
                final+=1
            else:
                values[parent]+=values[node]
            if(parent!=None and len(graph[parent])==1):
                q.append(parent)
    return final
```

<pre>If a leaf's value is divisible by k, we can safely separate it from the tree, thus, increasing the number of components. If not, it will be a part of its parent's component. To account for the latter, it is sufficient to just increase the parent's value by the leaf's value.

The algorithm proceeds by cutting leafs at each step and either cutting them (if they correspond to correct components) or merging them with parents (if not).</pre>


<h1>Find Building Where Alice and Bob Can Meet</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/description/">Click Here</a></p>

```python
def solve(n,heights,q,queries):
    arr=heights
    ans=[]
    for i in range(q):
        a,b=queries[i]
        if(a==b):
            ans.append(a)
            continue
        for j in range(max(a,b),n):
            if((arr[j]>arr[a]) and ((b==j and arr[j]>=arr[b]) or (arr[j]>arr[b]))) or ((arr[j]>arr[b]) and ((a==j and arr[j]>=arr[a]) or (arr[j]>arr[a]))):
                ans.append(j)
                break
        else:
            ans.append(-1)
    return ans

def optimized(n, arr):
    ans = [0] *n
    stack = []
    for i in range(n-1, -1, -1):
        while stack and arr[i] >= arr[stack[-1]]:
            stack.pop()
        if stack:
            ans[i]=(stack[-1])
        else:
            ans[i]=(-1)
        stack.append(i)
    return ans

def solve(n,arr,queries):
    ans=[]
    nge=optimized(n, arr)
    for a,b in queries:
        if(b<a):
            a,b=b,a
        if(a==b):
            ans.append(a)
            continue
        if(arr[a]<arr[b]):
            ans.append(b)
            continue
        if(nge[a]==-1 or nge[b]==-1):
            ans.append(-1)
            continue
        temp=nge[b]
        while temp!=-1 and arr[temp]<=arr[a]:
            temp=nge[temp]
        ans.append(temp)
    return ans
```

<p>Need to crack optimized solution yet</p>

<h1>Minimum Number of Operations to Sort a Binary Tree by Level</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/">Problem Link</a></p>

```python
def minSwaps(self, arr):
        mp = {}
        temp = sorted(arr)

        for i in range(len(arr)):
            mp[temp[i]] = i

        ans = 0
        i = 0
        while i < len(arr):
            ind = mp[arr[i]]
            if ind == i:
                i += 1
            else:
                arr[i], arr[ind] = arr[ind], arr[i]
                ans += 1
        return ans

def solve(root):
    queue=[root]
    ans=0
    while queue:
        arr=[]
        for i in range(len(queue)):
            node=queue.pop(0)
            if(node.left):
                queue.append(node.left)
                arr.append(node.left.val)
            if(node.right):
                queue.append(node.right)
                arr.append(node.right.val)
        ans+=minSwaps(len(arr),arr)
    return ans 
```

<p><strong>Solution : </strong><a href="https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/solutions/6176090/bfs-minswaps-simple-intuitive-approach">Click Here</a></p>

<h1>Find Largest Value in Each Tree Row</h1>

<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/">Problem Link</a></p>

```python
def solve(root):
    if not root:
        return []
    q=[root]
    ans=[root.val]
    while q:
        maxi=float('-inf')
        for i in range(len(q)):
            node=q.pop(0)
            if(node.left):
                q.append(node.left)
                maxi=max(maxi,node.left.val)
            if(node.right):
                q.append(node.right)
                maxi=max(maxi,node.right.val)
        if(q):
            ans.append(maxi)
    return ans
```



<h1>Target Sum</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/target-sum/description/">Click Here</a></p>

```python
def myfunc(n,pos,arr,target):
    if(pos==n):
        return 1 if (target==0) else 0
    include=myfunc(n,pos+1,arr,target-arr[pos])
    exclude=myfunc(n,pos+1,arr,target+arr[pos])
    return include+exclude

def myfunc(n,pos,arr,target,memo={}):
    key=(pos,target)
    if(pos==n):
        return 1 if (target==0) else 0
    if key in memo:
        return memo[key]
    include=myfunc(n,pos+1,arr,target-arr[pos],memo)
    exclude=myfunc(n,pos+1,arr,target+arr[pos],memo)
    memo[key]=include+exclude
    return include+exclude
```

<h1>Best Sightseeing Pair</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/best-sightseeing-pair/description/">Click Here</a></p>

```python
def solve(n,arr):
    maxi=0
    for i in range(n):
        for j in range(i+1,n):
            score=arr[i]+arr[j]+i-j
            maxi=max(maxi,score)
    return maxi

def solve(n,arr):
    suffix=[0]*n
    for i in range(n-1,-1,-1):
        if(i<n-1):
            suffix[i]=max(suffix[i+1],arr[i]-i)
        else:
            suffix[i]=arr[i]-i
    maxi=0
    for i in range(n-1):
        maxi=max(maxi,arr[i]+i+suffix[i+1])
    return maxi
```

<p><strong>Solution : </strong><a href="https://leetcode.com/problems/best-sightseeing-pair/solutions/6191106/only-suffix-array-o-n-100-beats/">Click Here</a></p>

<h1>Number of Ways to Form a Target String Given a Dictionary</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/">Click Here</a></p>

```python
from collections import defaultdict
Mod=(10**9)+7
def myfunc(n,target,pos,m,freq,index):
    if(pos==n):
        return 1
    if(index==m):
        return 0
    exclude=myfunc(n,target,pos,m,freq,index+1)
    cnt=freq[index][target[pos]]
    include=cnt*myfunc(n,target,pos+1,m,freq,index+1)
    return include+exclude

def myfunc(n,target,pos,m,freq,index,memo):
    if(pos==n):
        return 1
    if(index==m):
        return 0
    key=(pos,index)
    if key in memo:
        return memo[key]
    exclude=myfunc(n,target,pos,m,freq,index+1,memo)
    cnt=freq[index][target[pos]]
    include=cnt*myfunc(n,target,pos+1,m,freq,index+1,memo)
    memo[key]=(include+exclude)%Mod
    return memo[key]
```



<h1>Count Ways To Build Good Strings</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/count-ways-to-build-good-strings/description/">Click Here</a></p>

```python
def solve(low,high,zero,one,cnt):
    if(cnt>high):
        return 0
    includeZero=solve(low,high,zero,one,cnt+zero)
    includeOne=solve(low,high,zero,one,cnt+one)
    if(low<=cnt+zero<=high):
        includeZero+=1
    if(low<=cnt+one<=high):
        includeOne+=1
    return includeZero+includeOne

MOD=10**9+7
def solve(low,high,zero,one,cnt,memo):
    key=(cnt)
    if key in memo:
        return memo[key]
    if(cnt>high):
        return 0
    includeZero=solve(low,high,zero,one,cnt+zero,memo)
    includeOne=solve(low,high,zero,one,cnt+one,memo)
    if(low<=cnt+zero<=high):
        includeZero+=1
    if(low<=cnt+one<=high):
        includeOne+=1
    memo[key]=(includeZero+includeOne)%MOD
    return memo[key]



class Solution(object):
    def countGoodStrings(self, low, high, zero, one):
        """
        :type low: int
        :type high: int
        :type zero: int
        :type one: int
        :rtype: int
        """
        return solve(low,high,zero,one,0,{})
```

<h1>Minimum Cost For Tickets</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/minimum-cost-for-tickets/description/">Click Here</a></p>

```python
def myfunc(days,costs,pos,costInd,startInd):
    if(pos==len(days)):
        return 0
    include1,include7,include30=float('inf'),float('inf'),float('inf')
    if(costInd==0):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos)
        include7=costs[1]+myfunc(days,costs,pos+1,1,pos)
        include30=costs[2]+myfunc(days,costs,pos+1,2,pos)
    if(costInd==1):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos)
        include30=costs[2]+myfunc(days,costs,pos+1,2,pos)
        if(days[pos]-days[startInd]<7):
            include7=myfunc(days,costs,pos+1,1,startInd)
        else:
            include7=costs[1]+myfunc(days,costs,pos+1,1,pos)
    if(costInd==2):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos)
        include7=costs[1]+myfunc(days,costs,pos+1,1,pos)
        if(days[pos]-days[startInd]<30):
            include30=myfunc(days,costs,pos+1,2,startInd)
        else:
            include30=costs[2]+myfunc(days,costs,pos+1,2,pos)
    return min(include1,min(include7,include30))

def myfunc(days,costs,pos,costInd,startInd,memo):
    key=(pos,costInd,startInd)
    if key in memo:
        return memo[key]
    if(pos==len(days)):
        return 0
    include1,include7,include30=float('inf'),float('inf'),float('inf')
    if(costInd==0):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos,memo)
        include7=costs[1]+myfunc(days,costs,pos+1,1,pos,memo)
        include30=costs[2]+myfunc(days,costs,pos+1,2,pos,memo)
    if(costInd==1):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos,memo)
        include30=costs[2]+myfunc(days,costs,pos+1,2,pos,memo)
        if(days[pos]-days[startInd]<7):
            include7=myfunc(days,costs,pos+1,1,startInd,memo)
        else:
            include7=costs[1]+myfunc(days,costs,pos+1,1,pos,memo)
    if(costInd==2):
        include1=costs[0]+myfunc(days,costs,pos+1,0,pos,memo)
        include7=costs[1]+myfunc(days,costs,pos+1,1,pos,memo)
        if(days[pos]-days[startInd]<30):
            include30=myfunc(days,costs,pos+1,2,startInd,memo)
        else:
            include30=costs[2]+myfunc(days,costs,pos+1,2,pos,memo)
    memo[key]=min(include1,min(include7,include30))
    return memo[key]
```


<h1>Number of Ways to Split Array</h1>

<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/number-of-ways-to-split-array/description/">Click Here</a></p>

```python
def solve(n,arr):
    ans=0
    for i in range(n-1):
        leftSum=0
        rightSum=0
        for i in range(i+1):
            leftSum+=arr[i]
        for j in range(i+1,n):
            rightSum+=arr[j]
        if(leftSum>=rightSum):
            ans+=1
    return ans

def solve(n,arr):
    ans=0
    prefix=[]
    currentSum=0
    for i in range(n):
        currentSum+=arr[i]
        prefix.append(currentSum)
    for i in range(n-1):
        leftSum=prefix[i]
        rightSum=prefix[n-1]-prefix[i]
        if(leftSum>=rightSum):
            ans+=1
    return ans
```