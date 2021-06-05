# Nepenthe8's template

## 字符串处理

### 字符串哈希

~~~c++
typedef unsigned long long ull;
ull base = 131;
ull mod = 212370440130137957ll;
constexpr int maxn = 10010;

ull hashs(string s) { //var = hashs(string)
    ull ans = 0;
    for (int i = 0; i < s.length(); i++)
        ans = (ans * base + (ull)s[i]) % mod;
    return ans;
}
~~~

### 查询子串哈希值

~~~c++
typedef unsigned long long ull;
ull base = 131;

ull h1[N], h2[N], p[N]/* base的i次方 */;

ull hashs(int l, int r) { //假设要查询[i, j]的哈希值，调用hashs(i + 1, j + 1)，rhashs(i + 1, j + 1)
    return (h1[r] - h1[l - 1] * p[r - l + 1] % mod + mod) % mod;
}

ull rhashs(int l, int r) { /* 翻转子串的哈希值 */
    return (h2[l] - h2[r + 1] * p[r - l + 1] % mod + mod) % mod;
}

void hashs_init() { //为了防止查询时出现数组越界的情况，将整体的哈希值都向右移了一位
    p[0] = 1;
    h1[0] = (ull)s[0];
    for (int i = 1; i < s.length(); i++) {
        h1[i + 1] = ((h1[i] * base) % mod + (ull)s[i]) % mod;
        p[i] = (p[i - 1] * base) % mod; //预处理base的i次方
    }
    for (int i = s.length() - 1; i >= 0; i--) {
        h2[i + 1] = ((h2[i + 2] * base) % mod + (ull)s[i]) % mod;
    }
}
~~~

值得提一嘴的是，在查询子串哈希值时，不能使用自然溢出时采用的模数，因为要想取出区间[lf, rt]的值，那么我们需要在h[rt]这一位上，将h[lf - 1]累加的哈希值通过左移操作消除掉。而自然溢出无法保证得到的结果能够消除h[lf - 1]之前的影响。

### KMP

~~~c++
namespace KMP {
    int next[N];
    void kmp_pre(string x, int m) {
        int j = next[0] = -1;
        int i = 0;
        while (i < m) {
            while (-1 != j && x[i] != x[j])
                j = next[j];
            next[++i] = ++j;
        }
    }
    void preKMP(string x, int m) {
        int j = next[0] = -1; //初始化前缀末尾
        int i = 0; //初始化后缀末尾
        while (i < m) {
            while (-1 != j && x[i] != x[j]) //前后缀不相同
                j = next[j];
            if (x[++i] == x[++j]) next[i] = next[j]; //前后缀相同
            else next[i] = j;
        }
    }
    int kmp_count(string x, int m, string y, int n) { //x 是模式串，y 是文本串
        int j = 0, i = 0, ret = 0;
        preKMP(x, m);
        // kmp_pre(x, m);
        while (i < n) {
            while (-1 != j && y[i] != x[j])
                j = next[j];
            ++i, ++j;
            if (j >= m) {
                ++ret;
                j = next[j];
            }
        }
        return ret;
    }
}
using namespace KMP;
~~~


## 数学

### 埃氏筛

对于一个合数，肯定会被最小的质因子筛掉，那么对于当前质数$i$来说（合数的话它能筛掉的数已经被该合数的约数筛掉了），应该从$j=i \times i$开始筛，每轮$j = j + i$。

会出现重复筛的情况，比如12，会被2筛一次，被3筛一次。

时间复杂度：$O(nloglogn)$

~~~c++
constexpr int maxm = 10000010;
bool np[maxm]; //np[x] = 1表示x不是质数
void init(int n) {
    for (int i = 2; i * i <= n; i++) //对于一个合数，总有一个<=sqrt(n)的约数
        if (!np[i])
            for (int j = i * i; j <= n; j += i)
                np[j] = 1;
}
~~~


### 欧拉筛

时间复杂度：$O(n)$

~~~c++
constexpr int maxm = 10000010;
ll np[maxm], p[maxm], f[maxm], pn; //not prime(bool), prime[], f[i] is the smallest positive number m such that n/m is a square.

void Euler() {
    np[1] = 1;
    f[1] = 1;
    for (ll i = 2; i < maxm; i += 1) { //循环到maxm是为了把后面的数加入的质数表中
        if (not np[i]) {
            f[i] = i;
            p[pn++] = i; //质数表，下标从0开始
        }
        for (ll j = 0; j < pn; j += 1) {
            ll k = i * p[j];
            if (k >= maxm) break; //越界
            np[k] = 1; //标记合数
            if (f[i] % p[j]) f[k] = f[i] * p[j];
            else f[k] = f[i] / p[j];
            if (i % p[j] == 0) break; //当乘数是被乘数的倍数时，停止筛
        }
    }
}
~~~


## 数据机构

### 线段树

~~~c++
namespace SegTree {
    using T = long long;
    const int MAXN = 2e5 + 5, NA = 0; // NA是标记不可用时的值
    T tree[MAXN * 4], mark[MAXN * 4];
    int sN;
    T op(T a, T b) { return a + b; }
    void upd(int p, T d, int len) //p为当前节点标号，d为待更新的值，len为当前节点包含的区间长度
    {
        tree[p] += len * d;
        // (mark[p] == NA) ? (mark[p] = d) : (mark[p] += d);
        mark[p] += d;
    }

    template <class It>
    void build(It bg, int l, int r, int p)
    {
        if (l == r) { tree[p] = *(bg + l - 1); return; };
        int mid = (l + r) / 2;
        build(bg, l, mid, p * 2);
        build(bg, mid + 1, r, p * 2 + 1);
        tree[p] = op(tree[p * 2], tree[p * 2 + 1]);
    }
    template <class It>
    void build(It bg, It ed) // 这里的bg, ed是迭代器
    {
        sN = ed - bg;
        build(bg, 1, sN, 1);
    }

    void push_down(int p, int len)
    {
        if (mark[p] == NA) return;
        upd(p * 2, mark[p], len - len / 2);
        upd(p * 2 + 1, mark[p], len / 2);
        mark[p] = NA;
    }

    void update(int l, int r, T d, int p = 1, int cl = 1, int cr = sN) //[l, r]为查询的区间，[cl, cr]为当前节点包含的区间，p为当前节点标号，d为待更新的值
    {
        if (cl >= l && cr <= r) return upd(p, d, cr - cl + 1);
        push_down(p, cr - cl + 1);
        int mid = (cl + cr) / 2;
        if (mid >= l) update(l, r, d, p * 2, cl, mid);
        if (mid < r) update(l, r, d, p * 2 + 1, mid + 1, cr);
        tree[p] = op(tree[p * 2], tree[p * 2 + 1]);
    }

    T query(int l, int r, int p = 1, int cl = 1, int cr = sN)
    {
        if (cl >= l && cr <= r) return tree[p];
        push_down(p, cr - cl + 1);
        int mid = (cl + cr) / 2;
        if (mid >= r)
            return query(l, r, p * 2, cl, mid);
        else if (mid < l)
            return query(l, r, p * 2 + 1, mid + 1, cr);
        else
            return op(query(l, r, p * 2, cl, mid), query(l, r, p * 2 + 1, mid + 1, cr));
    }
} // namespace SegTree

using namespace SegTree;

~~~

### fhq_treap

~~~c++
namespace fhq_treap {
    #define getSZ(p) (p ? p->sz : 0)
    const int MAXN = 200010;
    struct Node {
        int key, rk;
        int sz; //记录以u为根节点的子树有多少节点
        Node *ls, *rs;
        void upd() {
            sz = getSZ(ls) + getSZ(rs) + 1; //左子+右子+自己
        }
    } pool[2 * MAXN]/* 节点池 */, *rt;
    int top; //指向节点池的指针
    void split(Node *p, Node *&pL, Node *&pR, int x) { //需要改变形态的二叉树参数需要传引用
        if (!p) {
            pL = pR = NULL;
            return;
        }
        if (p->key <= x) {
            pL = p;
            split(p->rs, pL->rs, pR, x);
            pL->upd();
        } else {
            pR = p;
            split(p->ls, pL, pR->ls, x);
            pR->upd();
        }
    }
    void merge(Node *&p, Node *pL, Node *pR) {
        if (!pL || !pR) { //如果某一个子树已经处理完了
            p = pL ? pL : pR;
            return;
        }
        if (pL->rk < pR->rk) {
            p = pL;
            merge(p->rs, pL->rs, pR);
        } else {
            p = pR;
            merge(p->ls, pL, pR->ls);
        }
        p->upd();
    }
    Node *newNode(int x) {
        Node *p = pool + (++top);
        p->key = x;
        p->rk = rand();
        p->sz = 1;
        return p;
    }
    void insert(Node *&rt, int x) {
        Node *p1, *p2;
        split(rt, p1, p2, x - 1);
        merge(rt, p1, newNode(x));
        merge(rt, rt, p2);
    }
    void remove(Node *&rt, int x) {
        Node *p1, *p2, *p3, *p4;
        split(rt, p1, p2, x - 1);
        split(p2, p3, p4, x);
        merge(p3, p3->ls, p3->rs);
        merge(p3, p3, p4);
        merge(rt, p1, p3);
    }

    int getRank(Node *&rt, int x) {
        Node *p1, *p2;
        split(rt, p1, p2, x - 1);
        int ret = getSZ(p1);
        merge(rt, p1, p2);
        return ret; //ret为比x小的数的个数
    }
    int kth(Node *p, int rk) {
        while (p) {
            if (rk <= getSZ(p->ls)) {
                p = p->ls;
            } else if (rk > getSZ(p->ls) + 1) { //当前要查找的rk > 左子树的节点数量 + 自己，说明在右子树之中
                rk -= getSZ(p->ls) + 1;
                p = p->rs;
            } else {
                return p->key;
            }
        }
        return 0;
    }
    int prev(Node *p, int x) {
        int ret = -INF;
        while (p) {
            if (x > p->key) {
                ckmax(ret, p->key);
                p = p->rs;
            } else {
                p = p->ls;
            }
        }
        return ret;
    }
    int next(Node *p, int x) {
        int ret = INF;
        while (p) {
            if (x < p->key) {
                ckmin(ret, p->key);
                p = p->ls;
            } else {
                p = p->rs;
            }
        }
        return ret;
    }
}
using namespace fhq_treap;
~~~

## 图论

### 堆优化Dijkstra

时间复杂度：$O((n + m)logm)$
vector的版本，在边数或点数过多的时候可能会MLE：

~~~c++
namespace Dijkstra {
    const ll maxn = 1000010;
    struct qnode {
        ll v;
        ll c;
        qnode(ll _v = 0, ll _c = 0) : v(_v), c(_c) {}
        bool operator<(const qnode &t) const {
            return c > t.c;
        }
    };
    struct Edge {
        ll v, cost;
        Edge(ll _v = 0, ll _cost = 0) : v(_v), cost(_cost) {}
    };
    vector<Edge> g[maxn];
    bool vis[maxn];
    ll dis[maxn];
    //点的编号从1开始
    void dijkstra(ll n, ll start) {
        for (ll i = 1; i <= n; i++) {
            vis[i] = false;
            dis[i] = INF;
        }
        priority_queue<qnode> q;
        while (!q.empty())
            q.pop();
        dis[start] = 0;
        q.push({start, 0});
        qnode t;
        while (!q.empty()) {
            t = q.top();
            q.pop();
            ll u = t.v;
            if (vis[u])
                continue;
            vis[u] = true;
            for (ll i = 0; i < g[u].size(); i++) {
                ll v = g[u][i].v;
                ll c = g[u][i].cost;
                if (!vis[v] && dis[v] > dis[u] + c) {
                    dis[v] = dis[u] + c;
                    q.push({v, dis[v]});
                }
            }
        }
    }
    void addedge(ll u, ll v, ll w) {
        g[u].push_back({v, w});
    }
}
using namespace Dijkstra;
~~~

用链式前向星：

~~~c++
namespace Dijkstra {
    const ll maxn = 1000010;
    struct qnode {
        ll v;
        ll c;
        qnode(ll _v = 0, ll _c = 0) : v(_v), c(_c) {}
        bool operator<(const qnode &t) const {
            return c > t.c;
        }
    };
    struct Edge {
        ll v, cost, next;
        Edge(ll _v = 0, ll _cost = 0, ll _next = 0) : v(_v), cost(_cost), next(_next) {}
    } g[maxn];
    // vector<Edge> g[maxn];
    ll head[N];
    bool vis[N];
    ll dis[N];
    //点的编号从1开始
    ll d_cnt = 0;
    void dijkstra(ll n, ll start) {
        for (ll i = 1; i <= n; i++) {
            vis[i] = false;
            dis[i] = INF;
        }
        priority_queue<qnode> q;
        while (!q.empty())
            q.pop();
        dis[start] = 0;
        q.push({start, 0});
        qnode t;
        while (!q.empty()) {
            t = q.top();
            q.pop();
            ll u = t.v;
            if (vis[u])
                continue;
            vis[u] = true;
            for (ll i = head[u]; ~i; i = g[i].next) {
                ll v = g[i].v;
                ll c = g[i].cost;
                if (!vis[v] && dis[v] > dis[u] + c) {
                    dis[v] = dis[u] + c;
                    q.push({v, dis[v]});
                }
            }
        }
    }
    void addedge(ll u, ll v, ll w) {
        g[d_cnt].v = v;
        g[d_cnt].cost = w;
        g[d_cnt].next = head[u];
        head[u] = d_cnt++;
    }
}
using namespace Dijkstra;
~~~

### SPFA

只要某个点u的dis[u]得到更新，并且此时不在队列中，就将其入队，目的是为了以u为基点进一步更新它的邻接点v的dis[v]。

这个是队列实现，有时候改成栈实现会更加快。

可以处理负权边和判定负环回路。

时间复杂度：$O(kE)$

~~~c++
namespace SPFA {
    const ll maxn = 1000010;
    struct Edge {
        ll v;
        ll cost;
        Edge(ll _v = 0, ll _cost = 0):v(_v), cost(_cost) {}
    };
    vector<Edge> g[maxn];
    void addedge(ll u, ll v, ll w) {
        g[u].push_back({v, w});
    }
    bool vis[maxn]; //在队列标志
    ll cnt[maxn]; //每个点的入队列次数
    ll dis[maxn];
    bool spfa(ll start, ll n) {
        for (ll i = 0; i <= n; i++) {
            vis[i] = false;
            dis[i] = INF;
            cnt[i] = 0;
        }
        vis[start] = true;
        dis[start] = 0;
        queue<ll> q;
        while (!q.empty())
            q.pop();
        q.push(start);
        cnt[start] = 1;
        while (!q.empty()) {
            ll u = q.front();
            q.pop();
            vis[u] = false;
            for (ll i = 0; i < g[u].size(); i++) {
                ll v = g[u][i].v;
                if (dis[v] > dis[u] + g[u][i].cost) {
                    dis[v] = dis[u] + g[u][i].cost;
                    //pre[v] = u; //表示v的前驱节点为u，在更新了权值之后紧接着更新pre数组
                    if (!vis[v]) {
                        vis[v] = true;
                        q.push(v);
                        if (++cnt[v] > n) //cnt[i]为入队列次数，用来判定是否存在负环回路
                            return false;
                    }
                }
            }
        }
        return true;
    }
}
using namespace SPFA;
~~~

### Floyd

解决任意两点间的最短路径的一种算法，可以正确处理有向图或负权的最短路径问题，同时也被用于计算有向图的传递闭包。Floyd-Warshall算法的时间复杂度为$O(N^3)$，空间复杂度为$O(N^2)$。

最外层循环相当于用k点作为中转点对全图进行更新。

~~~c++
namespace Floyd {
    constexpr int maxn = 1010;
    ll n, dis[maxn][maxn];
    void floyd() {
        for (int k = 1; k <= n; k++) {
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    ckmin(dis[i][j], dis[i][k] + dis[k][j]);
                }
            }
        }
    }
}
~~~

### 最小环

[HDU-1599](https://vjudge.gxu.mmmm.mn/problem/HDU-1599)

处理无向图上的最小环。

1. 一个环中编号最大的点为$k$，$i$和$j$是与$k$相邻的两个点，且满足$i<j<k$，则该环的长度为: ans = $d_{ij} + wt_{jk} + wt_{ki}$，$d_{ij}$是**不经过$k$点**的最短距离。
2. 使用floyd算法，当最外层枚举中转点到k时（尚未开始第$k$次循环），此时最路路径数组$d$还没有使用$k$更新过，那么此时$d_{ij}$就是未经过点k的最短距离。
3. 循环全部点，更新ans最小值。

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 210;
const ll INF = 0x3f3f3f3f3f3f3f3f;
template<class T> bool ckmin(T& a, const T& b) { return b < a ? a = b, 1 : 0; }

ll n, m, d[N][N]/* 最短路径 */, wt[N][N]/* 边的权值 */, ans;

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    while (cin >> n >> m) {
        ans = INF;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++) 
                if (i == j)
                    d[i][j] = wt[i][j] = 0;
                else
                    d[i][j] = wt[i][j] = INF;
        for (int i = 1, u, v, w; i <= m; i++) {
            cin >> u >> v >> w;
            if (w < d[u][v]) //防重边
                d[u][v] = wt[u][v] = w;
            if (w < d[v][u])
                d[v][u] = wt[v][u] = w;
        }
        for (int k = 1; k <= n; k++) {
            for (int i = 1; i < k; i++) { //保证i < j < k
                for (int j = i + 1; j < k; j++) {
                    ckmin(ans, d[i][j] + wt[j][k] + wt[k][i]);
                }
            }
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    ckmin(d[i][j], d[i][k] + d[k][j]);
                }
            }
        }
        if (ans == INF)
            cout << "It's impossible." << "\n";
        else
            cout << ans << "\n";
    }
    return 0;
}
~~~



## 搜索

## 动态规划

## 计算几何

## 其他
