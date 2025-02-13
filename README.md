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



<h1>Unique Length-3 Palindromic Subsequences</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/">Click Here</a></p>

```python
def solve(n,s):
    ans=set()
    def recursive(index,length,string):
        nonlocal ans
        if(length==3):
            if(string==string[::-1] and string not in ans):
                ans.add(string)
                return 1
            else:
                return 0
        if(index==n):
            return 0
        include=recursive(index+1,length+1,string+s[index])
        exclude=recursive(index+1,length,string)
        return include+exclude
    return recursive(0,0,"")

def solve(n,s):
    ans=set()
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                temp=s[i]+s[j]+s[k]
                if(temp==temp[::-1]):
                    ans.add(temp)
    return len(ans)

def solve(n,s):
    ans=set()
    prevIndex={}
    for i in range(n):
        if s[i] in prevIndex:
            for j in range(prevIndex[s[i]]+1,i):
                ans.add(s[i]+s[j]+s[i])
        if s[i] not in prevIndex:
            prevIndex[s[i]]=i
    return len(ans)
def solve(n,s):
    first=[None]*26
    last=[None]*26
    for i in range(n):
        if(first[ord('a')-ord(s[i])]==None):
            first[ord('a')-ord(s[i])]=i
        last[ord('a')-ord(s[i])]=i
    ans=0
    for i in range(26):
        if(first[i]!=None and last[i]!=None and first[i]<last[i]):
            ans+=len(set(s[first[i]+1:last[i]]))
    return ans
```

<h1>2381. Shifting Letters II</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/shifting-letters-ii/description/">Click Here</a></p>

```python
def nextChar(char,steps):
    steps%=26
    pos=ord(char)-ord('a')
    pos+=steps
    pos=pos%26
    return chr(ord('a')+pos)

def prevChar(char,steps):
    steps%=26
    pos=ord(char)-ord('a')
    pos-=steps
    if(pos<0):
        pos=26-abs(pos)
    return chr(ord('a')+pos)

def solve(n,s,shifts):
    s=list(s)
    for i,j,k in shifts:
        for index in range(i,j+1):
            if(k):
                s[index]=nextChar(s[index],1)
            else:
                s[index]=prevChar(s[index],1)
    return "".join(s)

def solve(n,s,shifts):
    s=list(s)
    totalShifts=[0]*n
    for i,j,k in shifts:
        if(k):
            totalShifts[i]+=1
            if(j<n-1):
                totalShifts[j+1]-=1
        else:
            totalShifts[i]-=1
            if(j<n-1):
                totalShifts[j+1]+=1
    for i in range(1,n):
        totalShifts[i]+=totalShifts[i-1]
    for i in range(n):
        if(totalShifts[i]<0):
            s[i]=prevChar(s[i],abs(totalShifts[i]))
        else:
            s[i]=nextChar(s[i],abs(totalShifts[i]))
    return "".join(s)
```


<h1>Minimum Number of Operations to Move All Balls to Each Box</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/description/">Click Here</a></p>

```python
def solve(n,s):
    ans=[]
    for i in range(n):
        cnt=0
        for j in range(i+1,n):
            if(s[j]=="1"):
                cnt+=(j-i)
        for j in range(0,i):
            if(s[j]=="1"):
                cnt+=(i-j)
        ans.append(cnt)
    return ans

def solve(n,s):
    ans=[]
    leftOps=[0]*n
    cnt=0
    if(s[0]=="1"):
        cnt+=1
    for i in range(1,n):
        leftOps[i]=leftOps[i-1]+cnt
        if(s[i]=='1'):
            cnt+=1
    rightOps=[0]*n
    cnt=0
    if(s[-1]=="1"):
        cnt+=1
    for i in range(n-2,-1,-1):
        rightOps[i]=rightOps[i+1]+cnt
        if(s[i]=='1'):
            cnt+=1
    for i in range(n):
        if(i==0):
            ans.append(rightOps[i])
        elif(i==n-1):
            ans.append(leftOps[i])
        else:
            ans.append(leftOps[i]+rightOps[i])
    return ans
```


<h1>Word Subsets</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/word-subsets/description/">Click Here</a></p>

```python
from collections import defaultdict
def solve(n,m,arr1,arr2):
    ans=[]
    count1=[0]*26
    for i in range(m):
        temp=[0]*26
        for j in arr2[i]:
            temp[ord(j)-ord('a')]+=1
            count1[ord(j)-ord('a')]=max(count1[ord(j)-ord('a')],temp[ord(j)-ord('a')])
    for i in range(n):
        count2=[0]*26
        for j in arr1[i]:
            count2[ord(j)-ord('a')]+=1
        for k in range(26):
            if(count1[k]>count2[k]):
                break
        else:
            ans.append(arr1[i])
    return ans
```

<h1>Construct K Palindrome Strings</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/construct-k-palindrome-strings/description/">Click Here</a></p>

```python
from collections import defaultdict
class Solution:
    def canConstruct(self, s: str, k: int) -> bool:
        n=len(s)
        if(k>n):
            return False
        if(k==n):
            return True
        count=defaultdict(int)
        for i in s:
            count[i]+=1
        oddCnt=0
        for i,v in count.items():
            if(v&1):
                oddCnt+=1
        if(oddCnt>k):
            return False
        return True
```

<p><strong>Solution</strong></p>
<p>The solution is based on the understanding that a string can be a palindrome only if it has at most 1 character whose frequency is odd. So if the number of characters having an odd frequency is greater than the number of palindromes we need to form, then naturally it's impossible to do so.</p>



<h1>Minimum Length of String After Operations</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/minimum-length-of-string-after-operations/description/">Click Here</a></p>

```python
def solve(n,s):
    s=list(s)
    for i in range(n):
        if(s[i]==""):
            continue
        a,b=None,None
        for j in range(i-1,-1,-1):
            if(s[j]==s[i]):
                a=j
                break
        for j in range(i+1,n):
            if(s[j]==s[i]):
                b=j
                break
        if(a!=None and b!=None):
            s[a]=""
            s[b]=""
    s="".join(s)
    return len(s)

def solve(n,s):
    count={i:s.count(i) for i in set(s)}
    deletions=0
    for i,cnt in count.items():
        deletions+=(cnt-2 if cnt%2==0 else cnt-1)
    return n-deletions
```


<h1>Bitwise XOR of All Pairings</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/bitwise-xor-of-all-pairings/description/">Click Here</a></p>

```python
class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        n1=len(nums1)
        n2=len(nums2)
        ans=0
        if(n1%2==0 and n2%2==0):
            ans=0
        elif(n1%2==0 and n2%2==1):
            for i in nums1:
                ans^=i
        elif(n1%2==1 and n2%2==0):
            for i in nums2:
                ans^=i
        elif(n1%2!=0 or n2%2!=0):
            for i in nums1:
                ans^=i
            for j in nums2:
                ans^=j
        return ans
```


<h1>Neighboring Bitwise XOR</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/neighboring-bitwise-xor/description/">Click Here</a></p>

```python
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        arr=derived
        n=len(arr)
        ans=0
        for i in range(n):
            ans^=arr[i]
        return bool(not ans)
```


<h1>Firstly completely painted Row or column</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/first-completely-painted-row-or-column/description/">Click Here</a></p>

```python
def solve(m,n,arr,matrix):
    row=[0]*m
    col=[0]*n
    index={}
    for r in range(m):
        for c in range(n):
            index[matrix[r][c]]=(r,c)
    for i in range(m*n):
        r,c=index[arr[i]]
        row[r]+=1
        col[c]+=1
        if(row[r]==n or col[c]==m):
            return i
```

<h1>Making Larger Island</h1>
<p><strong>Problem Statement :</strong><a href="https://leetcode.com/problems/making-a-large-island/description/?envType=daily-question&envId=2025-01-31">click here</a></p>

```python
def solve(n,matrix):
    visited=[[0]*n for i in range(n)]
    def bfs(n,i,j):
        nonlocal matrix
        nonlocal visited
        unique=f"{i}{j}"
        queue=[(i,j)]
        temp=[(i,j)]
        visited[i][j]=1
        maxi=0
        while queue:
            r,c=queue.pop()
            maxi+=1
            for i,j in zip([0,0,-1,1],[1,-1,0,0]):
                row,col=r+i,c+j
                if(row<n and col<n and row>=0 and col>=0 and matrix[row][col] and not visited[row][col]):
                    queue.append((row,col))
                    temp.append((row,col))
                    visited[row][col]=1
        
        for i,j in temp:
            matrix[i][j]=(maxi,unique)
        return maxi
    cnt=0
    for i in range(n):
        for j in range(n):
            if(matrix[i][j]==1):
                cnt+=1
            if(matrix[i][j]==1 and not visited[i][j]):
                temp=bfs(n,i,j)
    if(cnt==0):
        return 1
    ans=0
    print(matrix)
    for i in range(n):
        for j in range(n):
            temp=1
            check=[]
            if(matrix[i][j]==0):
                for k,l in zip([0,0,-1,1],[1,-1,0,0]):
                    r=i+k
                    c=j+l
                    if(0<=r<n and 0<=c<n and matrix[r][c] and matrix[r][c][1] not in check):
                       temp+=matrix[r][c][0]
                       check.append(matrix[r][c][1])
                       ans=max(ans,temp) 
    return n*n if(ans==0) else ans


class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        return solve(len(grid),grid)
```

<h1>Tuple with Same Product</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/tuple-with-same-product/description/"></a>Click Here</p>

```python
from itertools import permutations
from collections import defaultdict
from math import factorial
def solve(n,arr):
    cnt=0
    for a,b,c,d in permutations(arr,4):
        if(a*b==c*d):
            cnt+=1
    return cnt

def solve(n,arr):
    count=defaultdict(int)
    for i in range(n):
        for j in range(i+1,n):
            count[arr[i]*arr[j]]+=1
    ans=0
    print(count)
    for prod,cnt in count.items():
        if(cnt>1):
            ways=(factorial(cnt)//factorial(cnt-2))
            ans+=(ways*4)
    return ans


class Solution:
    def tupleSameProduct(self, nums: List[int]) -> int:
        arr=nums
        n=len(arr)
        return solve(n,arr)
```

<h1>Remove All Occurrences of a Substring</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/remove-all-occurrences-of-a-substring/">Click Here</a></p>

```python
class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        n=len(s)
        k=len(part)
        stack=[]
        i=0
        while i<n:
            temp=len(stack)
            if(len(stack)>=k and stack[temp-k:]==list(part)):
                stack=stack[:temp-k]
            stack.append(s[i])
            i+=1
        temp=len(stack)
        if(len(stack)>=k and stack[temp-k:]==list(part)):
            stack=stack[:temp-k]
        return "".join(stack)


```

<h1>Max Sum of a Pair With Equal Sum of Digits</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/">Click Here</a></p>


```python
from collections import defaultdict
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def sumOfDigits(n):
            sum=0
            while n>0:
                sum+=(n%10)
                n=n//10
            return sum
        sums=[]
        for i in nums:
            sums.append(sumOfDigits(i))
        maxi=-1
        prefix=defaultdict(int)
        for i in range(len(nums)):
            if sums[i] in prefix:
                maxi=max(maxi,nums[i]+prefix[sums[i]])
            prefix[sums[i]]=max(prefix[sums[i]],nums[i])
        return maxi


```

<h1>Clear Digits</h1>
<p><strong>Problem Link :</strong><a href="https://leetcode.com/problems/clear-digits/description/">Click Here</a></p>

```python
class Solution:
    def clearDigits(self, s: str) -> str:
        s=list(s)
        n=len(s)
        cnt=0
        for i in range(n-1,-1,-1):
            if(s[i].isdigit()):
                cnt+=1
                s[i]=""
            if(s[i].isalpha() and cnt>0):
                s[i]=""
                cnt-=1
        return "".join(s)
```

<h1>Minimum Operations to Exceed Threshold Value II</h1>
<p><strong>Problem Link : </strong><a href="https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/description/?">Click Here</a></p>

```python
import heapq

def solve(n,arr,k):
    arr.sort()
    a,b=arr[0],arr[1]
    del arr[0]
    del arr[0]
    n-=2
    cnt=0
    while a<k:
        arr.append(min(a,b)*2+max(a,b))
        cnt+=1
        arr.sort()
        n+=1
        if(n>=2):
            a,b=arr[0],arr[1]
            arr.pop(0)
            arr.pop(0)
            n-=2
        else:
            break
    return cnt

def solve(n,arr,k):
    heapq.heapify(arr)
    a,b=heapq.heappop(arr),heapq.heappop(arr)
    n-=2
    cnt=0
    while a<k:
        heapq.heappush(arr,min(a,b)*2+max(a,b))
        n+=1
        cnt+=1
        if(n>=2):
            a,b=heapq.heappop(arr),heapq.heappop(arr)
            n-=2
        else:
            break
    return cnt


class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        arr=nums
        n=len(arr)
        return solve(n,arr,k)
```