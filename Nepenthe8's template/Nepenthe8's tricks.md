# Nepenthe8's tricks

## 模型

### 柱形图最大矩阵面积

给出一个$n \times m$矩阵，要求从中找到每一列都是不下降数列的最大子矩阵，输出它的大小。

使用单调栈解决。[HDU-6957](https://vjudge.net/problem/HDU-6957)

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 2010;

ll n, m;
ll mp[N][N], h[N][N];

ll cal(ll h[]) {
    h[m + 1] = 0;
    stack<ll> sta;
    ll ans = 0;
    for (ll i = 1; i <= n; i++) {
        while (!sta.empty() && h[sta.top()] > h[i]) {
            ll now_h = h[sta.top()]; //栈顶矩形将会被弹出，以弹出的单个矩形条的高计算矩形面积
            sta.pop();
            ans = max(ans, ((i - 1) - (sta.empty() ? 1 : (sta.top() + 1)) + 1) * now_h); //确定左右边界(rt-lf+1)
        }
        sta.push(i);
    }
    while (!sta.empty()) {
        ll now_h = h[sta.top()];
        sta.pop();
        ans = max(ans, (m - (sta.empty() ? 1 : (sta.top() + 1)) + 1) * now_h); //这里的m为高度数组h的右端点位置
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int T; cin >> T;
    while (T--) {
        cin >> n >> m;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) {
                cin >> mp[i][j];
                if (mp[i][j] >= mp[i - 1][j])
                    h[i][j] = h[i - 1][j] + 1;
                else
                    h[i][j] = 1;
            }
        ll ans = 0;
        for (int i = 1; i <= n; i++)
            ans = max(ans, cal(h[i]));
        cout << ans << "\n";
    }
    return 0;
}
~~~

### 求每个元素产生的逆序对数量

[P3149 - 排序](https://www.luogu.com.cn/problem/P3149)

~~~c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll N = 300010;

struct Node {
    ll val, pos; //pos为第几个出现
    ll opt; //该点贡献的逆序对
    bool operator<(const Node& t) const {
        if (val == t.val) return pos < t.pos;
        else return val < t.val;
    }
} a[N];
ll tr[N], ta[N], atot;
ll n, m;

ll num[N]; //每个元素的逆序对贡献

void update(ll x, ll k) {
    for (; x <= n; x += x & -x)
        tr[x] += k;
}

ll query(ll x) {
    ll res = 0;
    for (; x; x -= x & -x) 
        res += tr[x];
    return res;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    //init
    // memset(tr, 0, sizeof(tr));
    // atot = 0;
    
    cin >> n >> m;
    for (ll i = 1; i <= n; i++) {
        cin >> a[i].val;
        a[i].pos = i;
        ta[++atot] = a[i].val;
    }

    sort(ta + 1, ta + 1 + atot);
    atot = unique(ta + 1, ta + 1 + atot) - ta - 1; //atot为离散后的表长

    for (ll i = n; i >= 1; i--) { //倒着求就可以算出每个点所产生的逆序对的数量
        a[i].val = lower_bound(ta + 1, ta + 1 + atot, a[i].val) - ta; //离散化，即原来的数->原来的数是第几小的
        a[i].opt = query(a[i].val - 1); //在他之后还比他小的 即 a[i].val贡献的逆序对数量
        // cout << a[i].opt << endl;
        num[i] = a[i].opt;
        update(a[i].val, 1);
    }
    for (int i = 1; i <= n; i++) {
        cout << num[i] << " \n"[i == n];
    }
    return 0;
}
~~~

### 运算式带括号的处理

本质上就是将给定的运算数任选两个进行运算，考虑暴力枚举出所有的运算数组合，即可认为是枚举出所有带括号的情况。

如[24dian](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=48558895)。

## STL

### 容器

#### deque

deque 是一种优化了的、对序列两端元素进行添加和删除操作的基本序列容器。它允许较为快速地随机访问，但它不像 vector 把所有的对象保存在一块连续的内存块，而是采用多个连续的存储块，并且在一个映射结构中保存对这些块及其顺序的跟踪。向 deque 两端添加或删除元素的开销很小。它不需要重新分配空间，所以向末端增加元素比 vector 更有效。

在两端进行push和pop的时间复杂度都为$O(1)$，随机访问的时间复杂度也为$O(1)$。

[参考](https://blog.csdn.net/like_that/article/details/98446479)

#### String

##### 构造函数

一些常见的构造函数形式

~~~c++
    string a; //定义一个空字符串
    string str_1 (str); //构造函数，全部复制
    string str_2 (str, 2, 5); //构造函数，从字符串str的第2个元素开始，复制5个（长度）元素，赋值给str_2
    string str_3 (ch, 5); //将字符串ch的前5个元素赋值给str_3
    string str_4 (5,'X'); //将 5 个 'X' 组成的字符串 "XXXXX" 赋值给 str_4
    string str_5 (str.begin(), str.end()); //复制字符串 str 的所有元素，并赋值给 str_5
~~~

##### erase() 和 substr()

substr() 没有参数为迭代器的重载

~~~c++
string str("0123456789");
cerr << str.substr(2, 4) << endl; //获得str中从2开始且长度为4的子串，输出2345
~~~

erase() 有整数或迭代器的重载

~~~c++
// 假定下面的操作都是相互独立的
//使用下标
str.erase(3);//从3开始到最后，输出012
str.erase(3, 5);//从3开始删除五个元素，即删除34567，输出01289

//使用迭代器
str.erase(str.begin() + 3);//删除位置3的元素，输出012456789
str.erase(str.begin() + 3,str.end());//删除位置3到末尾的元素（尾后迭代器之前），输出012
str.erase(str.begin() + 3, str.begin() + 8);//删除位置[3, 8)的元素，输出01289
~~~

### 函数

#### lower_bound()

返回第一个值大于等于x的数的下标。

##### 在降序序列中使用

如果需要在一个降序的序列$a$里面进行二分查找，可以考虑新建一个序列$b$，$b$中存放$a$对应位置的相反数，就可以继续使用lower_bound。

##### 在结构体或者vector中使用

如果需要在结构体或者vector进行lower_bound和upper_bound，把我们需要查找的数**封装成一个结构体**，才可以在结构体中进行查找。即使我们只需要针对某一维进行查找，也需要把整个结构体构造出来。

~~~c++
struct Node {
    int a, b;
    Node() {}
    Node(int a, int b) : a(a), b(b) {}
    bool operator<(const Node m) const { return b < m.b; } //定义比较方式，定义查找哪一维，使用重载运算符，lambda表达式报错
};

int main() {
    vector<Node> v;
    for (int i = 0; i < 10; i++) {
        v.push_back(Node(10 - i + 1, 2 * i));
        cout << v[i].a << "  " << v[i].b << endl;
    }
    int num; cin >> num;
    sort(v.begin(), v.end()); //进行二分之前需要排序
    int pos = lower_bound(v.begin(), v.end(), Node(0, num)) - v.begin(); //需要把我们查找的数封装成一个结构体才能进行查找。
    // int pos = lower_bound(v.begin(), v.end(), Node(0, num), [](const Node &A, const Node &B) { return A.b < B.b; }) - v.begin();
    cout << pos << endl;
    return 0;
}
~~~

##### 离散化

~~~c++
auto getRid = [&](int x) {
        return lower_bound(rows.begin(), rows.end(), x) - rows.begin() + 1;
    };
~~~

不加 1，表示取离散化后下标为 id 的元素；加1，表示取离散化后第id个元素。

#### __builtin_popcount()

__builtin_popcount()用于计算一个 32 位无符号整数有多少个位为1

- __builtin_popcount = int
- __builtin_popcountl = long int
- __builtin_popcountll = long long

#### partial_sum()与adjacent_difference()

partial_sum()用于求前缀和，adjacent_difference()用于求差分。

差分的每次修改都相当于对[L:]区间进行修改。

~~~c++
int a[5] = {15, 10, 6, 3, 1};
    int b[5] = {0};
    int c[5] = {0};
    adjacent_difference(a, a + 5, b); //15, -5, -4, -3, -2
    partial_sum(a, a + 5, c); //15, 25, 31, 34, 35
    for (int i = 0; i < 5; i++)
        cout << c[i] << ", ";
~~~

#### accumulate()

假设 vec 是一个 int 型的 vector 对象，下面的代码：

```c++
//sum the elements in vec starting the summation with the value 42
int sum = accumulate(vec.begin() , vec.end() , 42);
```

将 sum 设置为 vec 的元素之和再加上 42 。

accumulate 带有**三个形参**：头两个形参指定要**累加的元素范围**，第三个形参则是**累加的初值**。

accumulate 函数将它的一个内部变量设置为指定的初始值，然后在此初值上累加输入范围内所有元素的值。accumulate 算法返回累加的结果，**其返回类型就是其第三个实参的类型**。

用于指定累加起始值的第三个参数是必要的，因为 accumulate 对将要累加的元素类型一无所知，除此之外，没有别的办法创建合适的起始值或者关联的类型。

## 数学

### 注意

1. 模意义下出现减法时及时 + mod，遇到乘法就及时取 mod，以免后面要模多次造成 WA。

### 二进制拆分

一个数 $n$ 可以拆分为 $x$ 个数字，分别为：
$$
2^0,2^1,2^2,...,2^{k - 1},n-2^k+1, 其中k是满足n-2^k+1>0的最大整数
$$
满足使得这 $x$ 个数可以组合得到任意小于等于 $n$ 的正整数。

比如计算 $5$ 的二进制拆分，首先计算得到 $k$ 的取值，$k$ 是满足 $5 - 2^k + 1 > 0$ 的最大整数，得到 $k=2$，那么 $5$ 的二进制拆分得到的数字为 $2^0,2^1,5-2^2+1$，即 $1,2,2$，容易验证这三个数可以组合得到任意小于等于 $5$ 的正整数。

另外，$i$ 个 $2$ 的次幂刚好可以组合出所有二进制下 $1$ 的个数为 $i$  的数。

[p-binary](https://codeforces.com/problemset/problem/1247/C)就用到了二进制拆分。

化简下式子可以得到
$$
    n-i*p=\sum_{1}^{i} 2^k，k是任意自然数
$$

那么我们可以枚举 $i$，如果 $n - i * p$ 的二进制表示下$1$的数量等于 $i$，说明我们可以用 $i$ 个 $2$ 的幂次组合得到 $n - i * p$ 。注意枚举 $i$ 过程中 $i$ 的上限，等式右边的最小值，当且仅当 $k$ 全取 $0$，那么等式右边的最小值为 $i$，即如果 $n - i * p < i$，可以 break。

### 求和公式

平方和公式： $\sum_{k=1}^{n}k^2=1^2+2^2+3^2+...+n^2=\frac{n(n+1)(2n+1)}{6}$

立方和公式：$[\frac{(n \times (n + 1))}{2}]^2$

### 韦达定理

设一元二次方程 $ax^2+bx+c=0(a, b, c\in R, a \neq 0)$，两根 $x_1$、$x_2$ 有如下关系：
$$
x_1 + x_2 = - \frac{b}{a}
$$

$$
x_1 x_2 = \frac{c}{a}
$$

### 是否出现小数/分数

如出现小数/分数，将 double 转为 int 的过程中将会丢掉小数部分。

~~~c++
bool curF(double x) { //出现分数？
    return fabs(ll(x) - x) > eps;
}
~~~

### 同余定理

给定一个正整数 $m$，如果两个整数 $a$ 和 $b$ 满足 $a-b$ 能够被 $m$ 整除，即 $\frac {a-b}{m}$ 得到一个整数，那么就称整数 $a$ 与 $b$ 对模 $m$ 同余，记作 $a≡b$(mod m)。对模 $m$ 同余是整数的一个等价关系。

例题：[D - Integers Have Friends](https://codeforces.com/contest/1549/problem/D)。

## 计算几何

### 利用球面一般方程的系数计算球心坐标和半径

写出一般方程：$x^{2}+y^{2}+z^{2}+2ax+2by+2cz+d=0$

化作标准方程：$(x+a)^{2}+(y+b)^{2}+(z+c)^{2}=a^{2}+b^{2}+c^{2}-d$

易得球心坐标$O(-a,-b,-c)$，半径$r=\sqrt{a^{2}+b^{2}+c^{2}-d}$

在球面的一般方程中，如果将等号改为小于号，则表示球内部所有的点。

### 球缺

一个球被平面截下的一部分叫做球缺。截面叫做球缺的底面，垂直于截面的直径被截后，剩下的线段长叫做球缺的高。球缺曲面部分的面积（球冠面积）$S=2πRH$，球缺体积公式$V = \pi H^2(R- \frac{H}{3})$（$R$是球的半径,$H$是球缺的高)。

## 图论

### 注意

1. 使用邻接矩阵存图时，如题目无特殊说明，记得考虑重边、自环的情况

### 打印路径的两种方法

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

### 根节点的位置

在一棵树中，除根节点外，其余每个节点都有父节点（可以用来判断根节点的位置）

### 网络流

点的限制考虑拆点 https://vjudge.net/solution/32496101。

### 最短路

#### 用 Bellman Ford 求解有限制的最短路问题

「限制最多经过不超过 kk 个点」等价于「限制最多不超过 k + 1 条边」，而解决「有边数限制的最短路问题」是 SPFA 所不能取代 Bellman Ford 算法的经典应用之一（SPFA 能做，但不能直接做）。

Bellman Ford/SPFA 都是基于动态规划，其原始的状态定义为 $f[i][k]$ 代表从起点到 $i$ 点，且经过最多 $k$ 条边的最短路径。这样的状态定义引导我们能够使用 Bellman Ford 来解决有边数限制的最短路问题。

同样多源汇最短路算法 Floyd 也是基于动态规划，其原始的三维状态定义为 $f[i][j][k]$ 代表从点 $i$ 到点 $j$，且经过的所有点编号不会超过 $k$（即可使用点编号范围为 $[1, k]$）的最短路径。这样的状态定义引导我们能够使用 Floyd 求最小环或者求“重心点”（即删除该点后，最短路值会变大）。

[787. K 站中转内最便宜的航班 - 力扣（LeetCode）](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)

## 搜索

### 注意

1. dfs的中间变量，尤其是还要用于回溯的，老老实实定义成局部变量。比如在[天元突破 红莲螺岩](https://ac.nowcoder.com/acm/contest/16976/F)一题中，如果将dfs中临时记录的变量tmp定义成全局变量，那么在回溯的时候就会[出错](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=47957932)。
2. 在进行bfs时注意已经走过的点不可能再更新它的最小距离。
3. bfs时可以通过改变枚举方向的顺序使得在找到最短路径的情况下方向序列的字典序最小。如 [Penguins](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=48251717)。

~~~
/*(D<L<R<U)
     U
   L @ R
     D
     D
     D
 @LLLD
 
*/
~~~

4. 留意 dfs 的返回结果是否可以直接 return，还是放在 if 中作为条件判断。如[网格路径 - HDU 7037](https://vjudge.net/problem/HDU-7037)这题中， 如果直接 return dfs 的结果，可能由于找不到路径直接返回 false 了，实际上可以通过下面（下一行）的 dfs 可以找到一条路径，而由于在这一层递归中已经 return false 而造成答案错误。（就像 dfs 的匈牙利算法那样）

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
8. 模拟样例的时候尽量写出表达式，以便看出规律。按照模拟出来的表达式写代码，变量名不要误写成常数。（如[C - Sweets Eating](https://codeforces.com/contest/1253/problem/C)）
9. 有显然的递推式时可以考虑记忆化搜索或剪枝。（如[HDU - 6983](https://vjudge.net/problem/HDU-6983/origin)）
10. 关于异或的题牢牢抓住$ a \oplus a = 0$这个性质，考虑转换和抵消。
11. Σ式子化简时，可以展开之后再找规律。（[如](https://blog.csdn.net/qq_51354600/article/details/119638157)

#### 这为什么会WA呢？
1. 特判。（n=0, n=1?)
2. 答案是否是非负的，~~有可能在特判的时候神志不清~~。（比如[Chess Cheater](https://codeforces.com/contest/1427/submission/118950139)，在特判全为L的情况时使答案为-1，那么答案是不可能为负的，及时修改特判）
3. 初始化（省赛
4. 数组开小了（省赛
5. 爆int（做前缀和的时候开long long
6. 是否多组样例
7. 划分区间时特别注意最后一块（可能没有处理完，如[Silly Mistake](https://codeforces.com/contest/1253/problem/B))
8. 注意在更新答案的时候是不是在if中才更新，如果在if中是不是能够保证结果能够得到更新。（常见于dp中
9. 运行后直接死机注意是不是内存爆了的问题，比如关于vector的emplace_back函数死循环了。
10. 多组输入且需要memset的时候数组大小开的精确一些，不然有可能会TLE。

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
template <class T>
T randint(T l, T r = 0) // 生成随机数建议用<random>里的引擎和分布，而不是rand()模数，那样保证是均匀分布
{
    static mt19937 eng(time(0));
    if (l > r)
        swap(l, r);
    uniform_int_distribution<T> dis(l, r);
    return dis(eng);
}

//var = randint(1, 1000000);
//var = radint(1000000);
~~~

### function类模板

~~~c++
vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
    vector<vector<int>> ans;
    function<void(int, vector<int>, bitset<20>)> dfs = [&](int x, vector<int> now, bitset<20> vis) {
    // auto dfs = [&](int x, vector<int> now, bitset<20> vis) { //使用 auto 类型说明符声明的变量不能出现在其自身的初始值设定项中
        now.emplace_back(x);
        vis.set(x);
        if (x == int(graph.size()) - 1) {
            ans.emplace_back(now);
            return;
        }
        for (int i = 0; i < graph[x].size(); i++) {
            int v = graph[x][i];
            if (vis[v])
                continue;
            dfs(v, now, vis);
            vis.reset(x);
        }
        return;
    };
    bitset<20> bs;
    dfs(0, {}, bs);
    return ans;
}
~~~

