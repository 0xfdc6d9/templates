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

时间复杂度：$O(n\log \log n)$

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

### 矩阵快速幂

加速线性递推。

~~~c++
struct Matrix {
    ll mat[110][110];
    Matrix() { memset(mat, 0, sizeof(mat)); }
    Matrix operator*(const Matrix& b) const {
        Matrix res;
        for (ll i = 1; i <= n; i++) 
            for (ll j = 1; j <= n; j++)
                for (ll k = 1; k <= n; k++)
                    res.mat[i][j] = (res.mat[i][j] + mat[i][k] * b.mat[k][j]) % mod;
        return res;
    }
}ans, base;

void init() {
    for (ll i = 1; i <= n; i++)
        for (ll j = 1; j <= n; j++) {
            ll t; cin >> t;
            ans.mat[i][j] = base.mat[i][j] = t;
        }
}

void qpow(ll b) {
    while (b) {
        if (b & 1)
            ans = ans * base;
        base = base * base;
        b >>= 1;
    }
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n >> k;
    init();
    qpow(k - 1);
    for (ll i = 1; i <= n; i++) {
        for (ll j = 1; j <= n; j++) {
            cout << ans.mat[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}
~~~

### Berlekamp-Massey Algorithm

用于线性递推，时间复杂度为$O(n^2 \log m)$

[P5487 【模板】Berlekamp-Massey算法](https://www.luogu.com.cn/problem/P5487)

[A simple problem](https://ac.nowcoder.com/acm/contest/16976/A)

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1000000007; //998244353
ll powmod(ll a, ll b) { ll res = 1; a %= mod; assert(b >= 0); for (; b; b >>= 1) { if (b & 1) res = res * a % mod; a = a * a % mod; } return res; }

namespace Berlekamp_Massey {
    constexpr int MAXN = 1e4 + 7;
    #define rep(i, a, b) for (int i = a; i < b; i++)
    //              inital value
    vector<ll> BM(const vector<ll> &s)
    {
        int n = s.size(), L = 0, m = 0;
        vector<ll> C(n), B(n), T;
        C[0] = B[0] = 1;
        ll b = 1, d;
        for (int i = 0; i < n; i++)
        {
            m++;
            d = s[i];
            for (int j = 1; j <= L; j++)
                d = (d + C[j] * s[i - j]) % mod;
            if (!d)
                continue;
            T = C;
            ll coef = d * powmod(b, mod - 2) % mod;
            for (int j = m; j < n; j++)
                C[j] = (C[j] + mod - coef * B[j - m] % mod) % mod;
            if (2 * L > i)
                continue;
            L = i + 1 - L;
            B = T, b = d;
            m = 0;
        }
        C.resize(L + 1), C[0] = 0;
        for (ll &i : C)
            i = (mod - i % mod) % mod;
        return C;
    }
    void mul(vector<ll> &rec, ll a[], ll b[], int k)
    {
        ll c[MAXN] = {};
        rep(i, 0, k) rep(j, 0, k)
            c[i + j] = (c[i + j] + a[i] * b[j]) % mod;
        for (int i = k * 2 - 2; i >= k; i--)
            for (int j = 1; j <= k; j++) //use recursion to go back
                c[i - j] = (c[i - j] + rec[j] * c[i]) % mod;
        rep(i, 0, k) a[i] = c[i];
    } //   recursion   initial value   nth item
    ll linear(vector<ll> &a, vector<ll> &b, ll n)
    {
        int k = a.size() - 1;
        ll res[MAXN] = {}, c[MAXN] = {};
        c[1] = res[0] = 1;
        for (; n; n /= 2, mul(a, c, c, k))
            if (n & 1)
                mul(a, res, c, k);
        ll ret = 0;
        rep(i, 0, k) ret = (ret + b[i] * res[i]) % mod;
        for (int i = 1; i < a.size(); i++) { //最短线性递推式
            cout << a[i] << " \n"[i == a.size() - 1];
        }
        return ret;
    }
}
using namespace Berlekamp_Massey;

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    ll n, m; cin >> n >> m;
    vector<ll> F(n);
    for (int i = 0; i < n; i++) {
        cin >> F[i];
    }
    vector<ll> ans = BM(F);
    cout << linear(ans, F, m) << "\n";
    return 0;
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

#### 普通平衡树
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

在remove函数中，最后的merge写成
~~~c++
merge(p2, p3, p4);
merge(rt, p1, p2);
~~~
也可以AC

#### 文艺平衡树

~~~c++
namespace fhq_treap {
    #define getSZ(p) (p ? p->sz : 0)
    const int MAXN = 200010;
    struct Node {
        int key, rk;
        int sz; //记录以u为根节点的子树有多少节点
        Node *ls, *rs;
        bool rev;
        void upd() {
            sz = getSZ(ls) + getSZ(rs) + 1; //左子+右子+自己
        }
        void pushD() {
            if (!rev)
                return;
            if (ls) 
                ls->rev ^= 1;
            if (rs)
                rs->rev ^= 1;
            rev = 0;
            swap(ls, rs);
        }
    } pool[2 * MAXN]/* 节点池 */, *rt;
    int top; //指向节点池的指针
    void split(Node *p, Node *&pL, Node *&pR, int x) { //当fhq-treap用来【维护序列】时，split(p, pL, pR, k)函数的意义变为：将序列p的前k个元素分裂出来成一棵二叉树pL
        if (!p) {
            pL = pR = NULL;
            return;
        }
        p->pushD(); //每次操作前先下传标记
        if (getSZ(p->ls) + 1 <= x) {
            pL = p;
            split(p->rs, pL->rs, pR, x - (getSZ(p->ls) + 1));
            pL->upd();
        } else {
            pR = p;
            split(p->ls, pL, pR->ls, x);
            pR->upd();
        }
    }
    void merge(Node *&p, Node *pL, Node *pR) { //需要改变形态的二叉树参数需要传引用
        if (!pL || !pR) { //如果某一个子树已经处理完了
            p = pL ? pL : pR;
            return;
        }
        pL->pushD();
        pR->pushD();
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
    void reverse(Node *&rt, int l, int r) {
        Node *p1, *p2, *p3, *p4;
        split(rt, p1, p2, l - 1);
        split(p2, p3, p4, r - l + 1); //取出[l, r]的子树，根节点在p3
        p3->rev ^= 1; //在根节点打上rev标记
        merge(p2, p3, p4);
        merge(rt, p1, p2);
    }
    void midOrder(Node *p) { //中序遍历
        p->pushD();
        if (p->ls)
            midOrder(p->ls);
        cout << p->key << " ";
        if (p->rs)
            midOrder(p->rs);
    }
}
using namespace fhq_treap;
~~~

### 主席树

求静态区间第k小。

~~~c++
namespace hjt_tree {
    vector<ll> v;
    ll getid(ll x) { return lower_bound(v.begin(), v.end(), x) - v.begin() + 1; }
    struct Node { //每个点维护的是值域上值的个数
        ll lf, rt, sum; //该节点的左节点为hjt[lf]，右节点为hjt[rt]，值为sum
    } hjt[N * 40];
    ll count = 0, root[N];
    //pre的作用是now要依赖以上一个版本的权值线段树来建立
    void insert(ll l, ll r, ll pre, ll &now, ll p) {
        hjt[++count] = hjt[pre]; //等于上一个版本线段树的当前节点
        now = count;
        ++hjt[now].sum;
        if (l == r) return;
        ll m = (l + r) >> 1;
        if (p <= m) insert(l, m, hjt[pre].lf, hjt[now].lf, p);
        else insert(m + 1, r, hjt[pre].rt, hjt[now].rt, p);
    }
    //搜索到的当前节点所维护的区间为[l, r]
    //我们当前要查询[L, R]的权值线段树，Lnow表示L - 1版本的权值线段树遍历到的当前节点，Rnow表示R版本的权值线段树遍历到的当前节点
    ll query(ll l, ll r, ll Lnow, ll Rnow, ll kth) {
        if (l == r) return l;
        ll m = (l + r) >> 1;
        //决定向左半边递归还是右半边递归
        //看减出来的这个权值线段树的当前节点其左子树上有多少个数
        ll tmp = hjt[hjt[Rnow].lf].sum - hjt[hjt[Lnow].lf].sum;
        if (kth <= tmp) return query(l, m, hjt[Lnow].lf, hjt[Rnow].lf, kth); //都往左子树走
        else return query(m + 1, r, hjt[Lnow].rt, hjt[Rnow].rt, kth - tmp);
    }
}
using namespace hjt_tree;

ll n, m;
ll a[N];

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n >> m;
    for (ll i = 1; i <= n; i++) {
        cin >> a[i];
        v.push_back(a[i]);
    }
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    for (ll i = 1; i <= n; i++) {
        insert(1, n, root[i - 1], root[i], getid(a[i]));
    }
    for (ll i = 1, l, r, k; i <= m; i++) {
        cin >> l >> r >> k;
        cout << v[query(1, n, root[l - 1], root[r], k) - 1] << "\n";
    }
    return 0;
}
~~~

## 图论

### Dijkstra

#### 邻接矩阵形式

时间复杂度为$O(n^2)$，在涉及[边的增删](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=47959247)（虚）时比较方便。

~~~c++
namespace Dijkstra {
    const int MAXN = 1010;
    bool vis[MAXN];
    int pre[MAXN]; //pre[v] = u 表示v的前驱节点为u
    deque<pair<ll, ll> > road; //最短路径
    ll g[MAXN][MAXN], dis[MAXN];
    void dij_init(ll n) { /* 注意邻接矩阵的初始化 */
        for (ll i = 1; i <= n; i++)
            for (ll j = 1; j <= n; j++) 
                if (i == j)
                    g[i][j] = 0;
                else
                    g[i][j] = INF;
    }
    void dijkstra(ll n, ll start) {
        for (ll i = 1; i <= n; i++) {
            dis[i] = INF, vis[i] = false, pre[i] = -1;
        }
        dis[start] = 0;
        for (ll j = 1; j <= n; j++) {
            ll u = -1, minn = INF;
            for (ll i = 1; i <= n; i++) {
                if (!vis[i] && dis[i] < minn) {
                    minn = dis[i];
                    u = i;
                }
            }
            if (u == -1)
                break;
            vis[u] = true;
            for (ll v = 1; v <= n; v++) {
                if (!vis[v] && dis[u] + g[u][v] < dis[v]) {
                    dis[v] = dis[u] + g[u][v];
                    pre[v] = u;
                }
            }
        }
    }
    void dij_getRoad(ll end) { //传入终点，得到一条最短路径，存储在road中
        ll tmp = end;
        while (pre[tmp] != -1) {
            road.push_front({pre[tmp], tmp});
            tmp = pre[tmp];
        }
        // for (auto i : road) {
        //     cout << i.first << " " << i.second << "\n";
        // }
    }
}
using namespace Dijkstra;
~~~

#### 堆优化Dijkstra

时间复杂度：$O((n + m) \log m)$

##### 使用vector

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

##### 使用链式前向星

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

### 最大流

Dinic算法时间复杂度为$O(V^2E)$，在二分图中为$O(V \sqrt{E})$。

~~~c++
namespace Dinic {
    ll n, m, s, t, lv[N], cur[N]; // lv是每个点的层数，cur用于当前弧优化标记增广起点
    struct Node {
        ll to, w, next;
    } g[N];
    ll head[N], tot;

    inline void init() { //建边之前先初始化！
        memset(head, -1, sizeof(head));
    }

    void addedge(ll u, ll v, ll w) {
        g[tot].to = v;
        g[tot].w = w;
        g[tot].next = head[u];
        head[u] = tot++;

        g[tot].to = u;
        g[tot].w = 0;
        g[tot].next = head[v];
        head[v] = tot++;
    }

    bool bfs() // BFS分层
    {
        memset(lv, -1, sizeof(lv));
        lv[s] = 0;
        memcpy(cur, head, sizeof(head)); // 当前弧优化初始化
        queue<ll> q;
        q.push(s);
        while (!q.empty()) {
            ll p = q.front();
            q.pop();
            for (ll i = head[p]; ~i; i = g[i].next) {
                ll to = g[i].to, vol = g[i].w;
                if (vol > 0 && lv[to] == -1)
                    lv[to] = lv[p] + 1, q.push(to);
            }
        }
        return lv[t] != -1; // 如果汇点未访问过说明已经无法达到汇点，此时返回false
    }

    ll dfs(ll p = s, ll flow = INF)
    {
        if (p == t)
            return flow;
        ll rmn = flow; // 剩余的流量
        for (ll i = cur[p]; ~i && rmn; i = g[i].next) // 如果已经没有剩余流量则退出
        {
            cur[p] = i; // 当前弧优化，更新当前弧
            ll to = g[i].to, vol = g[i].w;
            if (vol > 0 && lv[to] == lv[p] + 1) // 往层数高的方向增广
            {
                ll c = dfs(to, min(vol, rmn)); // 尽可能多地传递流量
                rmn -= c; // 剩余流量减少
                g[i].w -= c; // 更新残余容量
                g[i ^ 1].w += c;
            }
        }
        return flow - rmn; // 返回传递出去的流量的大小
    }

    ll dinic() {
        ll ans = 0;
        while (bfs())
            ans += dfs();
        return ans;
    }
}
using namespace Dinic;
~~~

## 搜索

## 动态规划

### 多重背包

01背包是第$i$种物品可以取0件、取1件。

多重背包是第$i$种物品可以取0件、取1件、取2件、……、取$s_i$件。

那么我们可以将多重背包的问题转化为01背包求解：把第i种物品换成$s_i$件01背包的物品，每件物品的体积为$k*v_i$，价值为$k*w_i$。($0 \leq k \leq s_i$)。这样的时间复杂度为$O(m \sum s_i)$

#### 二进制优化

如果通过二进制拆分第$i$件物品，则可以转化成更少数量的01背包的物品，缩小问题规模。

二进制优化拆分物品数量$s$，$s$件拆分成$logs$件

时间复杂度为$O(m \sum \log s_i)$

[P1833 樱花](https://www.luogu.com.cn/problem/P1833)

~~~c++
pair<ll, ll> ts, te;
ll n, m, v[N], w[N], top, dp[N];

int main() {
    scanf("%lld:%lld %lld:%lld %lld", &ts.first, &ts.second, &te.first, &te.second, &n);
    m = te.first * 60 + te.second - ts.first * 60 - ts.second;
    for (ll i = 1, t, c, p; i <= n; i++) {
        scanf("%lld %lld %lld", &t, &c, &p);
        if (p == 0) {
            p = 1000000ll;
        }
        //二进制拆分，将第i种物品二进制拆分成若干件物品，每件物品的体积和价值乘一个拆分系数，就可以转化为01背包的物品求解
        /* 如体积为12，拆分系数为1，2，4，5，转化为4件01背包的物品，(v[i], w[i])、(2v[i], 2w[i])、(4v[i], 4w[i])、(5v[i], 5w[i]) */
        for (ll j = 1; j <= p; j <<= 1) {
            v[++top] = j * t;
            w[top] = j * c;
            p -= j;
        }
        if (p) { //剩余
            v[++top] = p * t;
            w[top] = p * c;
        }
    }
    for (ll i = 1; i <= top; i++) { //top为进行二进制拆分后物品的数量
        for (ll j = m; j >= v[i]; j--) {
            ckmax(dp[j], dp[j - v[i]] + w[i]);
        }
    }
    printf("%lld\n", dp[m]);
    return 0;
}
~~~

#### 单调队列优化

单调队列优化拆分的是背包容量$m$，根据体积$v$的余数，把$dp[0...m]$拆分成$v$个类，使$dp[0...m]$在$O(m)$内完成更新。

时间复杂度为$O(nm)$。

1. 对于滑动窗口范围的理解：$dp[j]$是由前面不超过数量$s$的同类值递推得到的。这就相当于从前面宽度为$s$的窗口挑选最大值来更新当前值。那么使用单调队列来维护窗口最大值，使更新$dp[j]$的次数缩减为1次。
2. $(k - q[h])/v*w$是还能放入的物品个数，其中$k - q[h]$为背包容量之差。
3. $dp[k]=$窗口中的max$+$还能放入物品的价值。
4. $dp[k]$通过属于上一个状态的$g[q[h]]$进行更新，所以窗口在$g$数组上滑动，也就是说$q$中存的是$g$数组的下标。
5. 判断队尾是否应该出队：若用g[k]比g[q[t]]更新后面f[x]能获得更大的价值，则下式成立（通过移项可以消掉x），队尾出队。

$$
g[k] + \frac {(x - k) * w} {v} \geq g[q[t]] + \frac {(x - q[t]) * w} {v}
$$

~~~c++
pair<ll, ll> ts, te;
ll n, m, dp[N], g[N], q[N];

int main() {
    scanf("%lld:%lld %lld:%lld %lld", &ts.first, &ts.second, &te.first, &te.second, &n);
    m = te.first * 60 + te.second - ts.first * 60 - ts.second;
    //多重背包 单调队列优化
    for (ll i = 1, v, w, s; i <= n; i++) {
        memcpy(g, dp, sizeof(dp)); //由于接下来体积从小到大枚举，所以将dp的上一个状态提前备份到g
        scanf("%lld %lld %lld", &v, &w, &s); //体积、价值、数量
        if (s == 0) { //表示属于完全背包的物品
            s = 1000010ll;
        }
        for (ll j = 0; j < v; j++) { //拆分成体积c个类
            ll h = 0, t = -1;
            for (ll k = j; k <= m; k += v) { //对每个类使用单调队列
                //q[h]不在窗口[k-s*v, k-v]内，队头出队
                if (h <= t && q[h] < k - s * v)
                    h++;
                //使用队头最大值更新
                if (h <= t)
                    dp[k] = max(g[k], g[q[h]] + (k - q[h]) / v * w);
                //当前值比队尾值更有价值，队尾出队
                while (h <= t && g[k] >= g[q[t]] + (k - q[t]) / v * w)
                    t--;
                //下标入队，使用下标是为了方便队头出队
                q[++t] = k;
            }
        }
    }
    printf("%lld\n", dp[m]);
    return 0;
}
~~~

### 数位dp

数位dp主要解决与数位有关的问题。

其基本的思想就是记忆化搜索保存搜索过的状态，通过从高位到低位暴力枚举能在每个数位上出现的数，搜索出某一区间$[L, R]$内的数，在搜索的过程中更新要求的答案。

比如求$[L, R]$区间内满足某种性质的数的数量，即计数问题时，我们先利用前缀和思想转化为求$[1, L - 1]$，$[1, R]$两个区间的问题，然后将数字按数位拆分出来，进行数位dp。

[ZJOI2010数字计数](https://www.luogu.com.cn/problem/P2602)

题意：给定两个正整数$a$和$b$，求在$[a, b]$中的所有整数中，每个数码(digit)各出现了多少次。

limit表示前面是否都贴着放。

lead表示前面是否都是前导0。

~~~c++
const int N = 66;
ll a[N];
ll dp[N][N][2][2];
ll len, nowd;

ll dfs(ll pos, ll cnt, bool limit, bool lead) {
    if (pos == len)
        return cnt;
    auto &d = dp[pos][cnt][limit][lead];
    if (d != -1)
        return d;
    ll mx = limit ? a[pos] : 9;
    ll ret = 0;
    for (ll i = 0; i <= mx; i++) {
        if (lead && i == 0)
            ret += dfs(pos + 1, cnt, limit && i == mx, 1);
        else 
            ret += dfs(pos + 1, cnt + (i == nowd), limit && i == mx, 0); //让他搜下去
    }
    d = ret;
    return ret;
}

ll solve(ll x) {
    memset(a, 0, sizeof(a));
    memset(dp, -1, sizeof(dp));
    len = 0;
    while (x) {
        a[len++] = x % 10;
        x /= 10;
    }
    reverse(a, a + len); //从高位到低位存储
    return dfs(0, 0, 1, 1);
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    ll a, b; cin >> a >> b;
    for (ll i = 0; i <= 9; i++) {
        nowd = i;
        cout << solve(b) - solve(a - 1) << " "; //要找[1, x]范围内满足条件的数
    }
    return 0;
}
~~~

[CF276D](https://codeforces.com/contest/276/submission/119204461)

题意：在区间$[L, R]$中找出两个数$a$、$b$$(a \leq b)$,使得$a \oplus b$达到最大。

$lb$表示是否贴着数位的下边界放、$rb$表示是否贴着数位的上边界放，这样可以保证枚举的数都是在$[L, R]$区间范围内。

二进制不太明显，我们考虑八进制数

~~~
1 2 3 4 5
      ^
      |
      i
假设枚举到第3位（0-index），x表示当前数位枚举的下界，y表示当前数位枚举的上界。
x = lb ? lf[3] : 0
y = lb ? rt[3] : 7(因为是8进制数)
那么接下来会枚举当前数位能填的数：
for (i = x; i <= y; i++)
	dfs(..., ..., lb && i == x, rb && i == y)
如果到i + 1位，lb标记为false，只可能是在第i位中填入了一个比下界大的数，那么从第i + 1位开始就可以任意填数，都会比下界的限制要大，即当前下界可以从0开始。
如果到i + 1位，rb标记为false，只可能是在第i位中填入了一个比上界小的数，那么从第i + 1位开始就可以任意填数，都会比上界的限制要小，即当前上界可以达到7。
~~~

可以看出，填入的数字保证在$[lf, rt]$区间内。

~~~c++
const int N = 66;
ll lf[N], rt[N], cnt, cnt1, cnt2, dp[N][2][2][2][2];

ll dfs(ll pos, bool lb0, bool rb0, bool lb1, bool rb1) {
    if (pos == cnt)
        return 0;
    auto &d = dp[pos][lb0][rb0][lb1][rb1];
    if (d != -1)
        return d;
    ll res = 0;
    for (ll i = lb0 ? lf[pos] : 0; i <= (rb0 ? rt[pos] : 1); i++) { //三目运算符的优先级比赋值号高，比比较运算符低
        for (ll j = lb1 ? lf[pos] : 0; j <= (rb1 ? rt[pos] : 1); j++) {
            ckmax(res, ((i ^ j) << (cnt - pos - 1)) + dfs(pos + 1, lb0 && i == lf[pos], rb0 && i == rt[pos], lb1 && j == lf[pos], rb1 && j == rt[pos]));
        }
    }
    d = res;
    return res;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    memset(dp, -1, sizeof(dp));
    ll a, b; cin >> a >> b;
    while (a) {
        lf[cnt1++] = a & 1;
        a >>= 1;
    }
    while (b) {
        rt[cnt2++] = b & 1;
        b >>= 1;
    }
    cnt = max(cnt1, cnt2);
    reverse(lf, lf + cnt);
    reverse(rt, rt + cnt);
    cout << dfs(0, true, true, true, true) << "\n";
    return 0;
}
~~~



## 计算几何

## 其他