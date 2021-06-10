# Nepenthe8's tricks

## 容器

### deque

deque 是一种优化了的、对序列两端元素进行添加和删除操作的基本序列容器。它允许较为快速地随机访问，但它不像 vector 把所有的对象保存在一块连续的内存块，而是采用多个连续的存储块，并且在一个映射结构中保存对这些块及其顺序的跟踪。向 deque 两端添加或删除元素的开销很小。它不需要重新分配空间，所以向末端增加元素比 vector 更有效。

在两端进行push和pop的时间复杂度都为$O(1)$，随机访问的时间复杂度也为$O(1)$。

[参考](https://blog.csdn.net/like_that/article/details/98446479)

## 数学

### 二进制拆分
一个数$n$可以拆分为$x$个数字，分别为：
$$
    2^0,2^1,2^2,...,2^{k - 1},n-2^k+1, 其中k是满足n-2^k+1>0的最大整数
$$
满足使得这$x$个数可以组合得到任意小于等于$n$的正整数。

比如计算$5$的二进制拆分，首先计算得到$k$的取值，$k$是满足$5 - 2^k + 1 > 0$的最大整数，得到$k=2$，那么$5$的二进制拆分得到的数字为$2^0,2^1,5-2^2+1$，即$1,2,2$，容易验证这三个数可以组合得到任意小于等于$5$的正整数。

另外，$i$个$2$的次幂刚好可以组合出所有二进制下 $1$的个数为$i$ 的数。

[p-binary](https://codeforces.com/problemset/problem/1247/C)就用到了二进制拆分。

化简下式子可以得到
$$
    n-i*p=\sum_{1}^{i} 2^k，k是任意自然数
$$

那么我们可以枚举$i$，如果$n - i * p$的二进制表示下$1$的数量等于$i$，说明我们可以用$i$个$2$的幂次组合得到$n - i * p$。注意枚举$i$过程中$i$的上限，等式右边的最小值，当且仅当$k$全取$0$，那么等式右边的最小值为$i$，即如果$n - i * p < i$，可以break。


## 图论

### 使用邻接矩阵存图时，如题目无特殊说明，记得考虑重边、自环的情况

### 打印路径的两种方法：

1. $pre[v] = u$，表示$v$的前驱为$u$，在更新了最小值之后紧接着更新$pre$数组。在打印结果时，从终点一路循环遍历到源点，将路过的节点依次储存在双端队列中。最后顺序遍历双端队列即可得到路径。

   ~~~c++
   ll tmp = end;
   while (pre[tmp] != -1) {
       road.push_front({pre[tmp], tmp});
       tmp = pre[tmp];
   }
   ~~~

   

2. $path[i][j] = k$，表示$i$到j的路径为$i$先到$k$，再从$k$到$j$。在打印结果时也可以通过while循环实现。初始化时$path[i][i] = i$，在每次松弛操作成功后更新$path$数组。

   ~~~c++
   printf("Path: %d", x);
   while (x != y) {
       printf("-->%d", path[x][y]);
       x = path[x][y];
   }
   ~~~

### 在一棵树中，除根节点外，其余每个节点都有父节点（可以用来判断根节点的位置）

## 搜索

### dfs的中间变量，尤其是还要用于回溯的，老老实实定义成局部变量

比如在[天元突破 红莲螺岩](https://ac.nowcoder.com/acm/contest/16976/F)一题中，如果将dfs中临时记录的变量tmp定义成全局变量，那么在回溯的时候就会[出错](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=47957932)。

### 记忆化搜索

[天元突破 红莲螺岩](https://ac.nowcoder.com/acm/contest/16976/F)，$dp[pos][lf][rt]$表示在$pos$位置，左边剩$lf$个，右边剩$rt$个时的最小花费。[code](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=47958190)

## 杂项

### 关于签到那些事

#### 签到的思路
1. 首先观察数据范围，确定是不是可以直接暴力/模拟/爆搜。
2. 如果没有很好的思路，手玩一下样例/打表看看是否存在规律。
3. 连续区间，考虑双指针、前缀和、差分。
4. 求最优解问题，先寻找局部最优解，确定使用贪心或dp。
5. 遇到构造题，先确定最终的目标是什么，慢慢往目标上靠。（比如[Matrix Problem](https://ac.nowcoder.com/acm/contest/16092/M)，需要让构造出来的矩阵具有很强的连通性，想到像手一样的模型）
6. 逆向思维有时会大大降低实现难度。
7. 考虑抽屉原理可以起到剪枝的效果。（比如[Shortest Cycle](https://codeforces.com/contest/1206/problem/D)，抽屉原理使得n的规模缩减，从而可以使用floyd算法）

#### 这为什么会WA呢？
1. 特判。（n=0, n=1?)
2. 答案是否是非负的，~~有可能在特判的时候神志不清~~。（比如[Chess Cheater](https://codeforces.com/contest/1427/submission/118950139)，在特判全为L的情况时使答案为-1，那么答案是不可能为负的，及时修改特判）
3. 初始化（省赛
4. 数组开小了（省赛
5. 爆int
6. 是否多组样例

### lower_bound
返回第一个值大于lower_bound

如果需要在一个降序的序列$a$里面进行二分查找，可以考虑新建一个序列$b$，$b$中存放$a$对应位置的相反数，就可以继续使用lower_bound

### 单调队列

~~~c++
//求滑动窗口的最大值
ll h = 0, t = -1;
for (ll i = 1; i <= n; i++) { //为了方便入队出队，单调队列中存储的是元素下标
    //如果队头q[h]已经不属于窗口[i-m+1, i]以内，则队头出队
    if (h <= t && q[h] < i - m + 1)
        h++;
    //如果待插入元素>=当前队尾，队尾出队（比你小还比你强
    while (h <= t && a[i] >= a[q[t]]) //如果求滑动窗口的最小值，则把>=改位<=
        t--;
    q[++t] = i;
    if (i > m - 1)
        cout << a[q[h]] << " ";
}
cout << "\n";
~~~

### 尺取法

给定长度为 $n$ 的数列整数 $a_0,a_1,…,a_{n-1}$以及整数 $S$。求出总和不小于 $S$ 的连续子序列的长度的最小值。如果解不存在，则输出 0。

~~~c++
ll n, S, a[N];

void solve() {
    ll res = n + 1;
    ll s = 0, t = 0, sum = 0;
    for (;;) {
        while (t < n && sum < S) { //移动下标的循环
            sum += a[t++];
        }
        if (sum < S) //退出尺取法搜索的条件
            break;
        res = min(res, t - s); //更新答案
        sum -= a[s++]; //更新左边界
    }
    if (res > n) { //解不存在
        res = 0;
    }
}
~~~

类似的题目：

[小幼稚买蛋糕](https://ac.nowcoder.com/acm/contest/16976/I)

~~~c++
ll n, x, ans, sum, v[N], pre[N];
 
int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n >> x;
    for (ll i = 0; i < n; i++) {
        cin >> v[i];
        if (i == 0)
            pre[i] = v[i];
        else
            pre[i] = pre[i - 1] + v[i];
    }
    ll lf = 0, rt = 0;
    for (;;) {
        while (rt < n && sum + v[rt] <= x) {
            sum += v[rt++];
            ans = max(ans, pre[rt - 1] - (lf - 1 < 0 ? 0 : pre[lf - 1]));
        }
        if (lf >= rt)
            break;
        sum -= v[lf++];
    }
    cout << ans << "\n";
    return 0;
}
~~~

### 随机数mt19937

周期长度：$2^{19937}-1$

产生的随机数的范围在int类型范围内。

也可以限制随机数的范围在$[lf, rt]$区间内。

~~~c++
#include <random>
const ll LF = 1;
const ll RT = 200000;
std::mt19937 gen(time(0));
std::uniform_int_distribution<> rnd(LF, RT);

//var = rnd(gen);
~~~