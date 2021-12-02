# Nepenthe8's template

[TOC]

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

值得提一嘴的是，在查询子串哈希值时，不能使用自然溢出时采用的模数，因为要想取出区间 $[lf, rt]$ 的值，那么我们需要在h[rt]这一位上，将 $h[lf - 1]$ 累加的哈希值通过左移操作消除掉。而自然溢出无法保证得到的结果能够消除 $h[lf - 1]$ 之前的影响。

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

### Z Algorithm

约定：字符串下标以0为起点。

对于一个长度为 $n$ 的字符串 $s$。定义函数 $z[i]$ 表示 $s$ 和 $s[i,n-1]$ （即以 $s[i]$ 开头的后缀）的最长公共前缀（LCP）的长度。$z$ 被称为 $s$ 的 $Z$ 函数。特别地，$z[0]=0$。

Z Algorithm 能够在 $O(n)$ 时间复杂度内计算 $Z$ 函数。

~~~c++
vector<int> z_function(string s) {
    int n = (int)s.length();
    vector<int> z(n);
    //z[0] = n;
    for (int i = 1, l = 0, r = 0; i < n; ++i) {
        if (i <= r && z[i - l] < r - i + 1) {
            z[i] = z[i - l];
        } else {
            z[i] = max(0, r - i + 1);
            while (i + z[i] < n && s[z[i]] == s[i + z[i]])
                ++z[i];
        }
        if (i + z[i] - 1 > r)
            l = i, r = i + z[i] - 1;
    }
    return z;
}
~~~

[P5410 【模板】扩展 KMP（Z 函数）](https://www.luogu.com.cn/problem/P5410)

可以访问 [这个网站](https://personal.utdallas.edu/~besp/demo/John2010/z-algorithm.htm) 来看 Z 函数的模拟过程。

### Trie

开的数组大小为字符集大小乘最大字符串长度。

#### 普通字典树

~~~c++
struct Trie {
    int tree[N][26], cnt; //从p号点指出的边为c(字符集中的一个字符)的下一个节点编号为 tree[p][c]
    int count[N]; //该结点出现的次数
    bool exist[N]; //该结点结尾的字符串是否存在
    Trie() {
        memset(exist, 0, sizeof(exist));
        memset(tree, 0, sizeof(tree));
        memset(count, 0, sizeof(count));
        cnt = 0;
    }
    void insert(string s, int l) {
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!tree[p][c]) //如果没有，就添加结点
                tree[p][c] = ++cnt;
            p = tree[p][c];
            ++count[p];
        }
        exist[p] = true;
    }
    bool find(string s, int l) {
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!tree[p][c])
                return false;
            p = tree[p][c];
        }
        return exist[p];
    }
    void del(string s, int l) {
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            p = tree[p][c];
            --count[p];
        }
        exist[p] = false;
    }
} trie;
~~~

#### 01字典树

常用于处理异或或者按位贪心问题。

[如](https://oj.lfengzheng.cn/problem/1482)：在给定的$N$个整数$A_1$，$A_2$，……，$A_n$中选出两个进行$xor$（异或）运算，求能得到的最大结果。

~~~c++
struct Trie {
    int nex[N][2], cnt; //从p号点指出的边为c(字符集中的一个字符)的下一个节点编号为nex[p][c]
    void insert(int x) {  // 插入字符串
        int p = 0;
        for (int i = 31; i >= 0; i--) {
            int c = x >> i & 1;
            if (!nex[p][c])
                nex[p][c] = ++cnt;  // 如果没有，就添加结点
            p = nex[p][c];
        }
    }
    int cal(int x) {
        int p = 0;
        int res = 0;
        for (int i = 31; i >= 0; i--) {
            int c = x >> i & 1;
            if (nex[p][!c]) {
                res += (1 << i);
                p = nex[p][!c];
            }
            else {
                p = nex[p][c];
            }
        }
        return res;
    }
}trie;

void solve() {
    read(n);
    for (int i = 1; i <= n; i++) {
        read(a[i]);
        trie.insert(a[i]);
    }
    for (int i = 1; i <= n; i++)
        ckmax(ans, trie.cal(a[i]));
    write(ans);
    putchar(10);
}
~~~

常用于处理区间异或问题。

如[HDU 6955. Xor Sum](http://acm.hdu.edu.cn/showproblem.php?pid=6955)

题意：给一个长度为$n$的一个整数序列$a_n$，寻找最短的，满足异或和大于等于$k$的连续子序列。输出子序列的左端点和右端点，若有多个最短长度的连续子序列，输出位置靠前的。不存在满足条件的连续子序列，输出−1。

~~~c++
int a[N], sum[N], mx[N]/* 经过trie树u节点的最靠右异或前缀和的下标 */;
int trie[N][2], val[N], tot;

void init() {
    trie[0][0] = trie[0][1] = 0;
    tot = 0;
    memset(mx, 0, sizeof(mx));
}

void insert(int x, int pos) {
    int u = 0;
    for (int i = 31; i >= 0; i--) {
        int v = (x >> i) & 1; //x在第i位上的数值
        ckmax(mx[u], pos);
        if (!trie[u][v]) {
            ++tot;
            trie[tot][0] = trie[tot][1] = val[tot] = 0; //init
            trie[u][v] = tot;
        }
        u = trie[u][v];
    }
    val[u] = x;
    ckmax(mx[u], pos);
}

//id为当前点的标号，sum表示当前搜索路径上两异或前缀和异或得到的区间异或和的前面一部分的大小，i为枚举到其二进制的位置
int dfs(int id, int sum, int i, int x, int k) {
    if (sum >= k) {
        return mx[id];
    } else if (sum + ((1ll << (i + 1)) - 1) < k) { //如果后面全1都无法大于等于k，剪枝
        return -1;
    }
    int res = -1;
    if (trie[id][0]) {
        ckmax(res, dfs(trie[id][0], sum + (x & (1 << i)), i - 1, x, k));
    }
    if (trie[id][1]) {
        ckmax(res, dfs(trie[id][1], sum + ((x & (1 << i)) ^ (1 << i)), i - 1, x, k));
    }
    return res;
}

int query(int x, int k) { //x为1-i的异或前缀和
    return dfs(0, 0, 31, x, k);
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int T; cin >> T;
    while (T--) {
        init();
        insert(0, 0);
        int n, k; cin >> n >> k;
        for (int i = 1; i <= n; i++) {
            cin >> a[i];
            sum[i] = sum[i - 1] ^ a[i];
        }
        int minLen = INF;
        int ans_lf = INF, ans_rt = -INF;
        for (int i = 1; i <= n; i++) {
            if (a[i] >= k) { //特判只有一个的情况
                minLen = 1;
                ans_lf = ans_rt = i;
                break;
            }
            int lpos = query(sum[i], k);
            if (lpos != -1 && (sum[lpos] ^ sum[i]) >= k) { //找到了异或和大于等于k的 离当前枚举的右端点 最近的左端点
                int len = i - lpos;
                if (len < minLen) { //长度小
                    minLen = len;
                    ans_lf = lpos + 1, ans_rt = i;
                } else if (len == minLen) { //长度小且最靠左
                    if (lpos + 1 < ans_lf) {
                        ans_lf = lpos + 1, ans_rt = i;
                    }
                }
            }
            insert(sum[i], i);
        }
        if (minLen == INF)
            cout << -1 << "\n";
        else
            cout << ans_lf << " " << ans_rt << "\n";
    }
    return 0;
}
~~~

#### 可持久化01trie

[luoguP4735 最大异或和](https://www.luogu.com.cn/problem/P4735)

给定一个非负整数序列 ${a}$，初始长度为$n$。

有 $m$ 个操作，有以下两种操作类型：

1. `A x`：添加操作，表示在序列末尾添加一个数 $x$，序列的长度 $n+1$。
2. `Q l r x`：询问操作，你需要找到一个位置 $p$，满足$l \le p \le r$，使得： $a[p] \oplus a[p + 1] \oplus ... \oplus a[N] \oplus x$ 最大，输出最大是多少。

~~~c++
struct Trie {
    static const int MAXN = 600010;
    int tot, rt[MAXN], ch[MAXN * 33][2], val[MAXN * 33]/* 出现次数 */;
    //从p号点指出的边为0/1的下一个节点编号为ch[p][c]
    int insert(int lst, int x) { //前一版本的根节点，添加的值
        int rt = ++tot, p = rt;
        for(int i = 28; i >= 0; i --) {
            int c = (x >> i) & 1;
            ch[p][c] = ++ tot, ch[p][c ^ 1] = ch[lst][c ^ 1]; //复制另一边的
            p = ch[p][c], lst = ch[lst][c]; 
            val[p] = val[lst] + 1;
        }
        return rt;
    }
    int query(int o1, int o2, int v) { //o1和o2版本
        int ret = 0;
        for (int i = 28; i >= 0; i--) {
            int t = (v >> i) & 1;
            if (val[ch[o1][!t]] - val[ch[o2][!t]]) //有出现过
                ret += (1 << i), o1 = ch[o1][!t], o2 = ch[o2][!t]; //尽量向不同的地方跳
            else
                o1 = ch[o1][t], o2 = ch[o2][t];
        }
        return ret;
    }
}st;

int n, m;
int a[N], pre[N];

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n >> m;
    st.rt[0] = st.insert(0, 0);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        pre[i] = pre[i - 1] ^ a[i];
    }
    for (int i = 1; i <= n; i++) {
        st.rt[i] = st.insert(st.rt[i - 1], pre[i]);
    }
    while (m--) {
        char op;
        int l, r, x;
        cin >> op;
        if (op == 'A') {
            ++n;
            cin >> a[n];
            pre[n] = pre[n - 1] ^ a[n];
            st.rt[n] = st.insert(st.rt[n - 1], pre[n]);
        } else if (op == 'Q') {
            cin >> l >> r >> x;
            l -= 2, --r;
            if (l < 0)
                cout << st.query(st.rt[r], 0, x ^ pre[n]) << "\n";
            else
                cout << st.query(st.rt[r], st.rt[l], x ^ pre[n]) << "\n"; //差分两个历史版本得到区间的trie
        }
    }
    return 0;
}
~~~

### AC-automaton

[AC 自动机 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/string/ac-automaton/)

时间复杂度 $O(\sum \left | s_i \right | + \left | S \right | )$，其中 $\left | s_i \right |$ 是模板串的长度，$\left | S \right |$ 是文本串的长度。

#### 基本概念

Trie 中的结点表示的是某个模式串的前缀。

AC 自动机的失配指针指向当前状态的最长后缀状态。

例子：ac-automaton1.gif 9号节点

$tr[u,c]$ 也可以理解为从状态（结点） 后加一个字符 `c` 到达的状态（结点），即一个状态转移函数 $trans[u, c]$

#### 关键点

在匹配字符串的过程中，我们会舍弃部分前缀达到最低限度的匹配。

fail 指针呢？它也是在舍弃前缀啊！试想一下，如果文本串能匹配 s，显然它也能匹配 s 的后缀。所谓的 fail 指针其实就是 s 的一个后缀集合。

例子：ac-automaton3.gif 9号节点

`tr` 数组还有另一种比较简单的理解方式：如果在位置 u 失配，我们会跳转到 fail[u] 的位置。所以我们可能沿着 fail 数组跳转多次才能来到下一个能匹配的位置。所以我们可以用 `tr` 数组直接记录记录下一个能匹配的位置，这样就能节省下很多时间。

这样修改字典树的结构，使得匹配转移更加完善。同时它将 fail 指针跳转的路径做了压缩（就像并查集的路径压缩），使得本来需要跳很多次 fail 指针变成跳一次。

~~~c++
#include <bits/stdc++.h>
using namespace std;
const int N = 155;
const int SZ = 155 * 85;

struct AC_automaton {
    static const int Num_CharacterSets = 26;
    int tr[SZ][Num_CharacterSets], tot, e[SZ], fail[SZ], cnt[N], val[SZ];
    void init() {
        memset(fail, 0, sizeof(fail));
        memset(tr, 0, sizeof(tr));
        memset(val, 0, sizeof(val));
        memset(cnt, 0, sizeof(cnt));
        memset(e, 0, sizeof(e));
        tot = 0;
    }
    void insert(string s, int id) { //trie 构建，id为模式串编号
        int u = 0; 
        int ns = s.length();
        s = " " + s;
        for (int i = 1; i <= ns; i++) {
            if (!tr[u][s[i] - 'a'])
                tr[u][s[i] - 'a'] = ++tot;
            u = tr[u][s[i] - 'a'];
        }
        e[u] = id; //标记该节点结尾的模式串编号
    }
    void build() {
        queue<int> q;
        while (!q.empty())
            q.pop();
        for (int i = 0; i < Num_CharacterSets; i++) //将根节点的子节点全部入队
            if (tr[0][i]) 
                q.push(tr[0][i]);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int i = 0; i < Num_CharacterSets; i++) {
                if (tr[u][i]) {
                    fail[tr[u][i]] = tr[fail[u]][i]; //fail指针指向当前状态的最长后缀状态
                    q.push(tr[u][i]);
                }
                else 
                    tr[u][i] = tr[fail[u]][i]; //修改字典树结构，将fail指针跳转的路径压缩
            }
        }
    }
    int query(string t) { //查询多模式串在文本串中最大匹配次数
        int nt = t.length();
        t = " " + t;
        int u = 0, res = 0;
        for (int i = 1; i <= nt; i++) {
            u = tr[u][t[i] - 'a'];
            for (int j = u; j; j = fail[j]) 
                val[j]++; //每匹配一个节点暴力跳fail到根，路上出现次数++
        }
        for (int i = 0; i <= tot; i++) {
            if (e[i]) { //如果有以i节点结尾的模式串，e[i]为原模式串编号，注意这里的if判断方式，模式串下标最好从1开始
                res = max(res, val[i]); //更新最大模式串匹配次数
                cnt[e[i]] = val[i]; //更新e[i]编号的模式串出现次数
            }
        }
        return res;
    }
} AC;

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n; 
    while (cin >> n) {
        if (!n) break;
        AC.init();
        string s[N], t; //s为模式串，t为文本串
        for (int i = 1; i <= n; i++) {
            cin >> s[i];
            AC.insert(s[i], i);
        }
        cin >> t;
        AC.build();
        int ans = AC.query(t);
        cout << ans << "\n";
        for (int i = 1; i <= n; i++) {
            if (AC.cnt[i] == ans) 
                cout << s[i] << "\n";
        }
    }
    return 0;
}
~~~

## 数学

### 素数筛

#### 埃氏筛

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


#### 欧拉筛

时间复杂度：$O(n)$。

~~~c++
constexpr int maxm = 10000010;
int p[maxm], pn; //prime[]
bool np[maxm]; //not prime(bool)

void Euler() {
    np[1] = 1;
    for (int i = 2; i < maxm; ++i) { //循环到maxm是为了把后面的数加入的质数表中，同时筛掉合数
        if (not np[i]) {
            p[pn++] = i; //质数表，下标从0开始
        }
        for (int j : p) {
            int k = i * j;
            if (k >= maxm) break; //越界
            np[k] = 1; //标记合数
            if (i % j == 0) break; //当乘数i是被乘数的倍数时，停止筛
        }
    }
}
~~~

#### miller_robin素数测试

时间复杂度为 $O(k \log^3n)$，$k$ 为底数选取的个数，$n$ 为待检验数。

由于这个过程中可能要计算 long long 型变量的平方，所以要考虑数据溢出的问题，解决方案是快速乘。

这里提供一个 $O(1)$ 的快速乘，原理是用溢出来解决溢出。

~~~c++
using ull = unsigned long long;
ll qmul(ll a, ll b, ll mod) //快速乘
{
    ll c = (long double)a / mod * b;
    ll res = (ull)a * b - (ull)c * mod;
    return (res + mod) % mod;
}
ll qpow(ll a, ll n, ll mod) //快速幂
{
    ll res = 1;
    while (n)
    {
        if (n & 1)
            res = qmul(res, a, mod);
        a = qmul(a, a, mod);
        n >>= 1;
    }
    return res;
}
const ll test_i64[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
const int test_i32[] = {2, 7, 61};
bool MRtest(ll n) //素数返回1，合数返回0
{
    if (n < 3 || n % 2 == 0)
        return n == 2; //特判
    ll u = n - 1, t = 0;
    while (u % 2 == 0)
        u /= 2, ++t;
    for (int a : test_i32)
    {
        ll v = qpow(a, u, n);
        if (v == 1 || v == n - 1 || v == 0)
            continue;
        for (int j = 1; j <= t; j++)
        {
            v = qmul(v, v, n);
            if (v == n - 1 && j != t)
            {
                v = 1;
                break;
            } //出现一个n-1，后面都是1，直接跳出
            if (v == 1)
                return 0; //这里代表前面没有出现n-1这个解，二次检验失败
        }
        if (v != 1)
            return 0; //Fermat检验
    }
    return 1;
}
~~~

### 分解质因数

#### 朴素算法

因数是成对分布的， n 的所有因数可以被分成两块，即 $[1, \sqrt{n}]$ ，和 $\sqrt{n} + 1, n]$。只需要把 $[1, \sqrt{n}]$ 里的数遍历一遍，再根据除法就可以找出至少两个因数了。

时间复杂度为 $O(\sqrt n)$。

~~~c++
struct PrimeFactor {
    ll num, tot;
} pfac[N]; //1-index
ll p_cnt; //记录素因数的种数
void pff(ll x) { //分解质因数
    for (ll i = 2; i * i <= x; i++) {
        while (x % i == 0) {
            x /= i;
            if (pfac[p_cnt].num == i) 
                pfac[p_cnt].tot++;
            else 
                pfac[++p_cnt].num = i, pfac[p_cnt].tot = 1;
        }
    }
    if (x > 1) {
        if (pfac[p_cnt].num == x) 
            pfac[p_cnt].tot++;
        else 
            pfac[++p_cnt].num = x, pfac[p_cnt].tot = 1;
    }
}
~~~

### 组合数

[例题](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=48654923&headNav=acm)

~~~c++
ll fac[N], inv[N];
void exgcd(int a, int b, ll &x, ll &y) //拓展欧几里得
{
    if (b == 0) {
        x = 1; y = 0;
        return;
    }
    exgcd(b, a % b, y, x);
    y -= a / b * x;
}
  
void init_C()
{
    fac[0] = 1;
    for (int i = 1; i < N; ++i)
        fac[i] = fac[i - 1] * i % mod; //阶乘数组
    ll x, y;
    exgcd(fac[N - 1], mod, x, y);
    inv[N - 1] = (x % mod + mod) % mod;
    for (int i = N - 2; i; --i) {
        inv[i] = inv[i + 1] * (i + 1) % mod; //逆元数组
    }
}
  
ll C(ll n, ll m)
{
    // assert(n >= m);
    if (n >= m)
        return 0;
    if (n == m || m == 0)
        return 1;
    return (fac[n] * inv[m] % mod * inv[n - m] % mod) % mod;
}
~~~

### 矩阵快速幂

加速线性递推。

例如，已知一个数列 a，它满足：
$$
a_{x}=\left\{\begin{array}{ll}
1 & x \in\{1,2,3\} \\
a_{x-1}+a_{x-3} & x \geq 4
\end{array}\right.
$$
求 $a$ 数列的第 $n$ 项对 $10^9+7$ 取余的值。

确定目标矩阵为：
$$
\begin{bmatrix}
 f[i] & f[i-1] & f[i-2]
\end{bmatrix}
$$
那么这个矩阵要怎样算出来。根据题目给出的递推式可以得到下面三个式子：
$$
\begin{array}{c}
f[i]=f[i-1] \times 1+f[i-2] \times 0+f[i-3] \times 1 \\
f[i-1]=f[i-1] \times 1+f[i-2] \times 0+f[i-3] \times 0 \\
f[i-2]=f[i-1] \times 0+f[i-2] \times 1+f[i-3] \times 0
\end{array}
$$
通过每一项的系数可以得出初始矩阵为：
$$
\begin{bmatrix}
 1 & 1 & 0 \\
 0 & 0 & 1 \\
 1 & 0 & 0
\end{bmatrix}
$$
然后我们就可以通过矩阵快速幂进行求解：
$$
\begin{bmatrix}
 f[i-1] & f[i-2] & f[i-3] 
\end{bmatrix}
*
\begin{bmatrix}
 1 & 1 & 0 \\
 0 & 0 & 1 \\
 1 & 0 & 0
\end{bmatrix}
=
\begin{bmatrix}
 f[i] & f[i-1] & f[i-2]
\end{bmatrix}
$$
第一个 ans 矩阵每左乘第二个 base 矩阵一次，使得矩阵内 第 $i$ 项变为第 $i + 1$ 项。

求数列第 $n$ 项的时间复杂度为 $O(\log n)$。

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1000000007; //998244353

const int order = 3; //矩阵的阶
struct Matrix { //1-index
    ll mat[order + 1][order + 1];
    Matrix() { memset(mat, 0, sizeof(mat)); }
    Matrix operator*(const Matrix& b) const {
        Matrix res;
        for (ll i = 1; i <= order; i++) 
            for (ll j = 1; j <= order; j++)
                for (ll k = 1; k <= order; k++)
                    (res.mat[i][j] += mat[i][k] * b.mat[k][j] % mod) %= mod;
        return res;
    }
} ans, base;

void matrixInit() {
    base.mat[1][1] = base.mat[1][2] = 1;
    base.mat[2][3] = base.mat[3][1] = 1;
    ans.mat[1][1] = ans.mat[1][2] = ans.mat[1][3] = 1;
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
    int T; cin >> T;
    while (T--) {
        memset(ans.mat, 0, sizeof(ans));
        memset(base.mat, 0, sizeof(base));
        ll n; cin >> n;
        if (n <= 3) {
            cout << 1 << "\n";
            continue;
        }
        matrixInit();
        qpow(n - 3);
        cout << ans.mat[1][1] << "\n";
    }
    return 0;
}

/* 
                            [ 1 1 0 ]
[ f[i-1] f[i-2] f[i-3] ]  * [ 0 0 1 ] = [ f[i] f[i-1] f[i-2] ]
                            [ 1 0 0 ]
 */
~~~

### Berlekamp-Massey Algorithm

用于线性递推，时间复杂度为$O(n^2 \log m)$，其中 $n$ 为数列 $P$ 的前 $n$ 项，$m$ 为要求的 $P_m$。

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

### 高斯消元

#### 列主元求解方程组

~~~c++
const int MAXN = 220;
double a[MAXN][MAXN], x[MAXN]; //方程的左边的矩阵和等式右边的值, 求解之后x存的就是结果, 0-index
int equ, var;                  //方程数和未知数个数
/*
* 返回0 表示无解,1 表示有解
*/
int Gauss() {
    int i, j, k, col, max_r;
    for (k = 0, col = 0; k < equ && col < var; k++, col++) {
        max_r = k;
        for (i = k + 1; i < equ; i++)
            if (fabs(a[i][col]) > fabs(a[max_r][col]))
                max_r = i;
        if (fabs(a[max_r][col]) < eps)
            return 0;
        if (k != max_r) {
            for (j = col; j < var; j++)
                swap(a[k][j], a[max_r][j]);
            swap(x[k], x[max_r]);
        }
        x[k] /= a[k][col];
        for (j = col + 1; j < var; j++)
            a[k][j] /= a[k][col];
        a[k][col] = 1;
        for (i = 0; i < equ; i++) {
            if (i != k) {
                x[i] -= x[k] * a[i][col];
                for (j = col + 1; j < var; j++)
                    a[i][j] -= a[k][j] * a[i][col];
                a[i][col] = 0;
            }
        }
    }
    return 1;
}

//                 系数       函数次数  自变量
double calf(vector<double> &p, int n, double x) { //p数组 0-index 依次表示(n - i - 1)次项的系数
    double ret = 0, pow_now = 1;
    for (int i = n; i >= 0; --i) {
        ret += pow_now * p[i];
        pow_now *= x;
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    vector<int> f = {1, 5, 15, 35, 70, 126, 210, 330, 495, 715, };
    int n = 7;
    equ = var = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = powmod(i + 1, n - j - 1);
        }
        x[i] = f[i];
    }
    Gauss();
    return 0;
}
~~~

#### 求解线性递推（模意义下

[Problem G. Pyramid](https://codeforces.com/gym/101981)

~~~c++
const int MAXN = 220;
ll a[MAXN][MAXN], x[MAXN]; //方程的左边的矩阵和等式右边的值, 求解之后x存的就是结果, 0-index
int equ, var;                  //方程数和未知数个数
/*
* 返回0 表示无解,1 表示有解
*/
int Gauss() {
    int i, j, k, col, max_r;
    for (k = 0, col = 0; k < equ && col < var; k++, col++) {
        max_r = k;
        for (i = k + 1; i < equ; i++)
            if (fabs(a[i][col]) > fabs(a[max_r][col]))
                max_r = i;
        if (fabs(a[max_r][col]) < eps)
            return 0;
        if (k != max_r) {
            for (j = col; j < var; j++)
                swap(a[k][j], a[max_r][j]);
            swap(x[k], x[max_r]);
        }
        (x[k] *= powmod(a[k][col], mod - 2)) %= mod;
        for (j = col + 1; j < var; j++)
            (a[k][j] *= powmod(a[k][col], mod - 2)) %= mod;
        a[k][col] = 1;
        for (i = 0; i < equ; i++) {
            if (i != k) {
                x[i] = (x[i] - (x[k] * a[i][col] % mod) + mod) % mod;
                for (j = col + 1; j < var; j++)
                    a[i][j] = (a[i][j] - (a[k][j] * a[i][col] % mod) + mod) % mod;
                a[i][col] = 0;
            }
        }
    }
    return 1;
}

//                 系数       函数次数  自变量
ll calf(vector<ll> &p, int n, ll x) { //p数组 0-index 依次表示(n - i - 1)次项的系数
    ll ret = 0, pow_now = 1;
    for (int i = n; i >= 0; --i) {
        (ret += pow_now * p[i] % mod) %= mod;
        (pow_now *= x) %= mod;
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    vector<int> f = {1, 5, 15, 35, 70, 126, 210, 330, 495, 715, };
    int n = 7;
    equ = var = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = powmod(i + 1, n - j - 1);
        }
        x[i] = f[i];
    }
    Gauss();
    return 0;
}
~~~



### FFT

~~~c++
const int N = 1e6 + 7;
typedef complex<double> CP;
const int lim = 1 << 21; //limit << 1，这里提前算出了limit的值为20
const double Pi = acos(-1);
const int OFFSET = 500007;
CP a[lim], b[lim];
bool vis[lim];

void FFT(CP *x, int lim, int inv) // 板子而已
{
    int bit = 1, m;
    CP stand, now, temp;
    while ((1 << bit) < lim)
        ++bit;
    for (int i = 0; i < lim; ++i)
    {
        m = 0;
        for (int j = 0; j < bit; ++j)
            if (i & (1 << j))
                m |= (1 << (bit - j - 1));
        if (i < m)
            swap(x[m], x[i]);
    }
    for (int len = 2; len <= lim; len <<= 1)
    {
        m = len >> 1;
        stand = CP(cos(2 * Pi / len), inv * sin(2 * Pi / len));
        for (CP *p = x; p != x + lim; p += len)
        {
            now = CP(1, 0);
            for (int i = 0; i < m; ++i, now *= stand)
            {
                temp = now * p[i + m];
                p[i + m] = p[i] - temp;
                p[i] = p[i] + temp;
            }
        }
    }
    if (inv == -1)
        for (int i = 0; i < lim; ++i)
            x[i].real(x[i].real() / lim);
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n, m; cin >> n >> m;
    for (int i = 0; i < n + 1; i++) {
        int x; cin >> x;
        a[i].real(x);
    }
    for (int i = 0; i < m + 1; i++) {
        int x; cin >> x;
        b[i].real(x);
    }
    

    int limit = 1; //把长度补到2的幂，不必担心高次项的系数，因为默认为0
    while (limit <= n + m)
        limit <<= 1;
    FFT(a, limit, 1);
    FFT(b, limit, 1);
    for (int i = 0; i < limit; ++i) {
        a[i] *= b[i];
    }
    FFT(a, limit, -1);

    for (int i = 0, x; i <= n + m; i++) { //枚举结果多项式的指数
        x = (int)floor(a[i].real() + 0.5); //x为每一项的系数
        cout << x << " \n"[i == limit];
    }
    return 0;
}
~~~

### NTT

~~~c++
const int MAXN = 3 * 1e6 + 7;
ll L, r[MAXN];
const int P = 998244353, G = 3, Gi = 332748118, OFFSET = 500001;
ll limit = 1;
ll powmod(ll a, ll b, ll mod) { ll res = 1; a %= mod; assert(b >= 0); for (; b; b >>= 1) { if (b & 1) res = res * a % mod; a = a * a % mod; } return res; }
void NTT(ll *A, int type) {
    for (int i = 0; i < limit; i++)
        if (i < r[i])
            swap(A[i], A[r[i]]);
    for (int mid = 1; mid < limit; mid <<= 1) {
        ll Wn = powmod(type == 1 ? G : Gi, (P - 1) / (mid << 1), P);
        for (int j = 0; j < limit; j += (mid << 1)) {
            ll w = 1;
            for (int k = 0; k < mid; k++, w = (w * Wn) % P) {
                ll x = A[j + k], y = w * A[j + k + mid] % P;
                A[j + k] = (x + y) % P,
                      A[j + k + mid] = (x - y + P) % P;
            }
        }
    }
}

int n, m;
ll a[MAXN], b[MAXN];

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n >> m;
    for (int i = 0; i < n + 1; i++) { //0-index
        int x; cin >> x;
        a[i] = x; //给指数为i的系数赋值为x
    }
    for (int i = 0; i < m + 1; i++) {
        int x; cin >> x;
        b[i] = x;
    }
    
    while (limit <= n + m)
        limit <<= 1, L++;
    for (int i = 0; i <= limit; i++)
        r[i] = (r[i >> 1] >> 1) | (i & 1) << (L - 1);
    NTT(a, 1);
    NTT(b, 1);
    for (int i = 0; i <= limit; i++)
        a[i] = (a[i] * b[i]) % P;
    NTT(a, -1);
    ll inv = powmod(limit, P - 2, P);

    for (int i = 0; i <= n + m; i++) {
        int x = a[i] * inv % P;
        cout << x << " \n"[i == n + m];
    }
    return 0;
}
~~~

### 数论分块

整除分块应用于类似如下的这个式子：
$$
\sum_{i = 1}^{n} ⌊\frac{n}{i}⌋
$$
存在$O (\sqrt n)$的解法。

打表可以发现$\frac{n}{i}$的值呈现块状分布，且对于任意一个块，它的最后一个数都在$n/(n/i)$位置上。

于是可以这样$O (\sqrt n)$的处理：

~~~cpp
for (int l = 1, r; l <= n; l = r + 1) {
    r = n / (n / l); //每一块最后一个数的位置
    ans += (r - l + 1) * (n / l); //每一块的个数*每一块的值
    cout << ans << endl;
}
~~~

例题：[F - Fair Distribution](https://codeforces.com/gym/103055/problem/F)。

### BSGS

大步小步算法（baby step giant step，BSGS）是一种用来求解离散对数（即模意义下对数）的算法，即给出 $a^x \equiv b (\mod{m})$ 中 $a$，$b$，$m$的值（这里保证 $a$ 和 $m$ 互质），求解 $x$ 。

~~~c++
ll BSGS(ll a, ll b, ll m)
{
    static unordered_map<ll, ll> hs;
    hs.clear();
    ll cur = 1, t = sqrt(m) + 1;
    for (ll B = 1; B <= t; ++B)
    {
        (cur *= a) %= m;
        hs[b * cur % m] = B; // 哈希表中存B的值
    }
    ll now = cur; // 此时cur = a^t
    for (ll A = 1; A <= t; ++A)
    {
        auto it = hs.find(now);
        if (it != hs.end())
            return A * t - it->second;
        (now *= cur) %= m;
    }
    return -1; // 没有找到，无解
}
~~~

例题：[HDU-6956](https://vjudge.net/problem/HDU-6956)。

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

[HDU-7116](https://vjudge.net/problem/HDU-7116)

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pii = pair<ll, ll>;
const double eps = 1e-6;
const int N = 2e5 + 7;
// const int N = 7;
const int INF = 0x3f3f3f3f;
const int mod = 998244353; //998244353
const int dir[8][2] = {0, 1, 0, -1, 1, 0, -1, 0,/* dir4 */ -1, -1, -1, 1, 1, -1, 1, 1};
ll gcd(ll a, ll b) { return b == 0 ? a : gcd(b, a % b); }
ll powmod(ll a, ll b) { ll res = 1; a %= mod; assert(b >= 0); for (; b; b >>= 1) { if (b & 1) res = res * a % mod; a = a * a % mod; } return res; }
template <class T> bool ckmin(T &a, const T &b) { return b < a ? a = b, 1 : 0; }
template <class T> bool ckmax(T &a, const T &b) { return a < b ? a = b, 1 : 0; }
template <class T> void debug(const string &Name, const T &a) { cerr << "# " << Name << ": " << a << endl; }
template <class A, class... B> void debug(const string &Name, const A &a, const B &...b) { cerr << "# " << Name << ": " << a << endl; debug(b...); }

inline ll lowbit(ll x) { return x & -x; }

inline int ls(int x) { return x << 1; }
inline int rs(int x) { return x << 1 | 1; }
struct Node {
    int l, r;
    ll x, sum;
    int lazy;
    int flag;
} tree[N << 2];

void pushD(int p) {
    if (tree[p].lazy == 0)
        return;
    (tree[ls(p)].sum *= powmod(2, tree[p].lazy)) %= mod;
    (tree[rs(p)].sum *= powmod(2, tree[p].lazy)) %= mod;
    tree[ls(p)].lazy += tree[p].lazy;
    tree[rs(p)].lazy += tree[p].lazy;
    tree[p].lazy = 0;
}

void pushUp(int p) {
    tree[p].flag = tree[ls(p)].flag & tree[rs(p)].flag;
    (tree[p].sum = tree[ls(p)].sum + tree[rs(p)].sum) %= mod;
}

void buildT(int l, int r, int p = 1) {
    tree[p].l = l, tree[p].r = r;
    tree[p].sum = tree[p].lazy = tree[p].flag = 0;
    if (l == r) {
        cin >> tree[p].x;
        tree[p].sum = tree[p].x;
        if (tree[p].x == lowbit(tree[p].x))
            tree[p].flag = 1;
        return;
    }
    int mid = (l + r) >> 1;
    buildT(l, mid, ls(p));
    buildT(mid + 1, r, rs(p));
    pushUp(p);
}

void upd(int l, int r, int p = 1) { //[l, r]为查询区间
    int cl = tree[p].l, cr = tree[p].r;
    if (l <= cl && cr <= r && tree[p].flag) { //区间都可以直接乘2
        (tree[p].sum *= 2) %= mod;
        ++tree[p].lazy;
        return;
    }
    if (cl == cr) { //暴力修改，最多log次
        tree[p].x = tree[p].sum = tree[p].x + lowbit(tree[p].sum); //叶子节点才用x，存a[i]，用于判断它是否能够直接乘2
        if (tree[p].x == lowbit(tree[p].x))
            tree[p].flag = 1;
        return;
    }
    pushD(p);
    int mid = (cl + cr) >> 1;
    if (l <= mid)
        upd(l, r, ls(p));
    if (mid + 1 <= r)
        upd(l, r, rs(p));
    pushUp(p);
}

ll query(int l, int r, int p = 1) {
    int cl = tree[p].l, cr = tree[p].r;
    if (l <= cl && cr <= r) {
        return tree[p].sum;
    }
    pushD(p);
    ll ans = 0;
    int mid = (cl + cr) >> 1;
    if (l <= mid)
        (ans += query(l, r, ls(p))) %= mod;
    if (mid + 1 <= r)
        (ans += query(l, r, rs(p))) %= mod;
    return ans;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int T; cin >> T;
    while (T--) {
        int n; cin >> n;
        buildT(1, n);
        int m; cin >> m;
        while (m--) {
            int op, l, r; cin >> op >> l >> r;
            if (op == 1) 
                upd(l, r);
            else 
                cout << query(l, r) << "\n";
        }
    }
    return 0;
}
~~~



### 树状数组

~~~c++
struct BIT {
    BIT () { memset(tr, 0, sizeof(tr)); }
    ll n, tr[N];
    void update(ll x, ll k) {
        for (; x <= n; x += x & -x) { //注意n的取值
            tr[x] += k;
        }
    }
    ll query(ll x) {
        ll res = 0ll;
        for (; x; x -= x & -x) {
            res += tr[x];
        }
        return res;
    }
};
~~~

### fhq_treap

#### 普通平衡树

【treap】 是一种弱平衡的二叉搜索树。treap 这个单词是 tree 和 heap 的组合，表明 treap 是一种由树和堆组合形成的数据结构。treap 的每个结点上除了关键字 key 之外，还要额外储存一个值 rk (priority)。treap 除了要满足关于**key的BST性质（左子<根<右子，即拍扁序是有序的）之外，还需满足关于rk的小根堆性质**。而 rk 是每个结点建立时随机生成的，因此 treap 是**期望平衡（即高度=logn）**的。

- split(p, pL ,pR, x) 将 **二叉树 p 分裂成 2 棵树 pL 和 pR ，其中 pL 包含所有 <=x 的结点，pR 包含剩下的节点**。

- merge(p, pL, pR) 将 **二叉树 pL 和 pR 重新合并成一棵树 p，注意此处要求 pL 和 pR 的值域是不重叠（没有交叉）的**。由于树的高度是期望log的，split&merge函数的复杂度都是**O(logn)**。并且分裂之后的 pL、pR、以及合并之后的p也同时满足heap性质，它们也是期望平衡的。

由于树的高度是期望log的，split&merge函数的复杂度都是**O(logn)**。并且分裂之后的 pL、pR、以及合并之后的p也同时满足heap性质，它们也是期望平衡的。

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
            pL = pR = NULL; //截断不属于这棵树的部分
            return;
        }
        if (p->key <= x) {
            pL = p; //先整个复制过来，接着继续修改多余的部分
            split(p->rs, pL->rs, pR, x);
            pL->upd();
        } else {
            pR = p;
            split(p->ls, pL, pR->ls, x);
            pR->upd();
        }
    }
    void merge(Node *&p, Node *pL, Node *pR) {
        if (!pL || !pR) { //如果某一个子树已经处理完了，直接把剩余的部分接上去
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

当fhq-treap用来【维护序列】时，split(p, pL, pR, k)函数的意义变为：**将序列 p 的前k个元素分裂出来成一棵二叉树 pL** ，写法类似。

fhq-treap可以通过split操作把序列中的任意一段[l,r]分裂出来，可以方便地在上面打各种标记。

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
    void split(Node *p, Node *&pL, Node *&pR, int x) {
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

维护值域：询问有关数的大小，如第几大。

维护序列：询问关于位置，如区间[l, r]，第pos位。

如果不幸需要debug：空指针、引用（需要改变树形态都需要传引用）、指针该不该指。

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

### 莫队

莫队可以解决一类 **离线区间询问** 问题，适用性极为广泛。同时将其加以扩展，便能处理树上路径询问（树上莫队）以及支持修改（带修莫队）操作。

假设 **区间长度n** 与 **询问数量m** 同阶，那么对于序列上的区间询问问题，如果从 [l,r] 的答案能够 $O(1)$ 扩展到 [l+1,r]/[l-1,r]/[l,r-1]/[l,r+1]（即支持 O(1) insert/remove 维护信息）的答案，那么可以在 $O(N\sqrt N)$ 的复杂度内求出所有询问的答案。

做法：所有询问离线后排序，顺序处理每个询问，暴力从上一个区间的答案转移到下一个区间答案。

莫队的~~玄学~~优化：

1. 奇偶性优化（因为顺序发生了变化所以调试起来不是很方便）
2. 移动指针的常数压缩
3. 块长
4. int/long long

#### 普通莫队

~~~c++
ll pos[N];
//以左端点 lf 所在块的编号 pos[lf] = lf/B 为第一关键字，右端点 rt 为第二关键字从小->大排序。
struct inq {
    ll lf, rt, id;
    bool operator<(const inq &t) const {
        if (pos[lf] != pos[t.lf])
            return pos[lf] < pos[t.lf];
        else return rt < t.rt;
    }
} q[N];

ll n, m, a[N], ans[N], nowAns;

void update(ll pos, ll sign) { //O(1)
    
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n;
    ll B = sqrt(n);
    cin >> m;
    for (ll i = 1; i <= n; i++) {
        cin >> a[i];
        pos[i] = i / B;
    }
    for (ll i = 0; i < m; i++) {
        cin >> q[i].lf >> q[i].rt;
        q[i].id = i;
    }
    sort(q, q + m);
    for (ll i = 0, l = 1, r = 0; i < m; i++) {
        while (l < q[i].lf) update(l++, -1);
        while (l > q[i].lf) update(--l, +1); //+1为要加入到当前维护的区间中
        while (r < q[i].rt) update(++r, +1);
        while (r > q[i].rt) update(r--, -1);
        ans[q[i].id] = nowAns;
    }
    for (ll i = 0; i < m; i++) {
        
    }
    return 0;
}
~~~

### ST表

以区间最值为例，设 $f[i][j]$ 表示当前区间的 $[i, i + 2^j - 1]$ 内的最值，显然 $f[i][0] = max[i][i] = num_i$ 。

由倍增思想可得，跳 $2^i$ 步相当于先跳 $2^{i - 1}$ 步再跳 $2^{i - 1}$ 步；同理区间 $[i, i+2^j - 1]$ 内的最值相当于是区间 $[i, i + 2^{j - 1} - 1]$ 和 $[i + 2 ^ {j - 1}, i + 2^j - 1]$ 内的最值。

所以可得式子 $f[i][j] = max(f[i][j - 1], f[i + 2^{j - 1}][j - 1])$ 。

则只需要枚举起点（也就是枚举 $i$ ），接着枚举区间长度（也就是枚举 $j$ ），使得整个区间被包进去，就可以构建出ST表了。

ST表常用于解决**可重复贡献问题**。常见的可重复贡献问题有：区间最值、区间按位和、区间按位或、区间GCD等。而像区间和这样的问题就不是可重复贡献问题。

ST表预处理的时间复杂度为$O(n \log n)$，查询的时间复杂度为$O(1)$。

为了[卡常](https://ac.nowcoder.com/acm/contest/11256/K)，一般会调换 $i$、$j$ 两个维度。

~~~c++
class SparseTable {
public:
    int lg[N] = {-1};
    ll st[24][N];
    template <class T>
    T op(T &a, T &b) { return max(a, b); } //检查区间操作！
    SparseTable() {
        for (int i = 1; i < N; i++) {
            lg[i] = lg[i / 2] + 1;
        }
    }
    inline void init(int n) {
        //完成初始化！ for i in [1, n]: st[0][i] = val[i]
        for (int i = 1, x; i <= n; i++) {
            cin >> x; st[0][i] = x;
        }
        build(n);
    }
    inline void build(int n) {
        for (int i = 1; i <= lg[n]; i++)
            for (int j = 1; j + (1 << i) - 1 <= n; j++) 
                st[i][j] = op(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
    }
    ll query(int l, int r) {
        ll len = lg[r - l + 1];
        return op(st[len][l], st[len][r - (1 << len) + 1]);
    }
};

SparseTable ST;

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n, m; cin >> n >> m;
    ST.init(n);
    while (m--) {
        int l, r; cin >> l >> r;
        cout << ST.query(l, r) << "\n";
    }
    return 0;
}
~~~



## 图论

### 并查集

同时使用路径压缩和启发式合并之后，并查集的每个操作平均时间仅为 $O(\alpha(n))$ ，其中 $\alpha$ 为阿克曼函数的反函数，其增长极其缓慢，也就是说其单次操作的平均运行时间可以认为是一个很小的常数。

~~~c++
class Dsu {
public:
    static const int MAXN = 2e5 + 7;
    int fa[MAXN], rk[MAXN];
    void init(int n) {
        for (int i = 1; i <= n; i++) {
            fa[i] = i, rk[i] = 1;
        }
    }
    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
    void merge(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) {
            return;
        }
        if (rk[x] >= rk[y]) {
            fa[y] = x;
        } else {
            fa[x] = y;
        }
        if (rk[x] == rk[y] && x != y)
            rk[x]++;
    }
    bool isSame(int x, int y) { return find(x) == find(y); }
} dsu;
~~~

#### 拓展域并查集

如果我们需要记录连通块的 size 的话，由于并不知道每个人具体属于哪一类，只是知道他肯定属于其中一类，那么在初始化 size 时只用将其中一类赋值为1即可。

~~~c++
#include "bits/stdc++.h"
using namespace std;
using ll = long long;

/*
四类动物的食物链，A 吃 B，B 吃 C, C 吃 D，D 吃 A
第一种说法是 1 X Y，表示 X 和 Y 是同类。
第二种说法是 2 X Y，表示 X 吃 Y。
第三种说法是 3 X Y，表示 X 和 Y 不是同类，且 X 不吃 Y，Y 也不吃 X。
 */

class Dsu {
public:
    static const int MAXN = 3e5 + 7;
    int fa[MAXN], rk[MAXN];
    void init(int n) {
        for (int i = 1; i <= n; i++) {
            fa[i] = i, rk[i] = 1;
        }
    }
    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
    void merge(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) {
            return;
        }
        if (rk[x] >= rk[y]) {
            fa[y] = x;
        } else {
            fa[x] = y;
        }
        if (rk[x] == rk[y] && x != y)
            rk[x]++;
    }
    bool isSame(int x, int y) { return find(x) == find(y); }
} dsu;

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n, k; cin >> n >> k;
    dsu.init(4 * n);
    int ans = 0;
    while (k--) {
        int opt, x, y; cin >> opt >> x >> y;
        if (x > n || y > n) {
            ++ans;
            continue;
        }
        if (opt == 1) {
            if (dsu.isSame(x, y + n) || dsu.isSame(x, y + 3 * n) || dsu.isSame(x, y + 2 * n)) { //如果有非同类关系，则表明不是同类
                ++ans;
            } else { //合并同类
                dsu.merge(x, y);
                dsu.merge(x + n, y + n);
                dsu.merge(x + 2 * n, y + 2 * n);
                dsu.merge(x + 3 * n, y + 3 * n);
            }
        } else if (opt == 2) {
            if (dsu.isSame(x, y) || dsu.isSame(x, y + 3 * n) || dsu.isSame(x, y + 2 * n)) { //如果有非捕食关系，则表明是假话
                ++ans;
            } else { //构建捕食关系
                dsu.merge(x, y + n);
                dsu.merge(x + n, y + 2 * n);
                dsu.merge(x + 2 * n, y + 3 * n);
                dsu.merge(x + 3 * n, y);
            }
        } else {
            if (dsu.isSame(x, y) || dsu.isSame(x, y + n) || dsu.isSame(x, y + 3 * n)) { //如果有同类、捕食、被捕食关系，则说明是假话
                ++ans;
            } else { //构建 非同类 且 非捕食 且 非被捕食 关系
                dsu.merge(x, y + 2 * n);
                dsu.merge(x + n, y + 3 * n);
                dsu.merge(x + 2 * n, y);
                dsu.merge(x + 3 * n, y + n);
            }
        }
    }
    cout << ans << "\n";
    return 0;
}
~~~

### 最短路

#### Dijkstra

Dijkstra 是基于一种贪心的策略，首先用数组 dis 记录起点到每个结点的最短路径，再用一个数组保存已经找到最短路径的点。

然后，从 dis 数组选择最小值，则该值就是源点 s 到该值对应的顶点的最短路径，并且把该点记为已经找到最短路。

此时完成一个顶点，再看这个点能否到达其它点（记为 v ），将 dis[v] 的值进行更新。

不断重复上述动作，将所有的点都更新到最短路径。

这种算法实际上是 $O(n^2)$ 的时间复杂度，但我们发现在 dis 数组中选择最小值时，我们可以用 STL 里的堆来进行优化，堆的一个性质就是可以在 nlogn 的时限内满足堆顶是堆内元素的最大（小）值。

##### 邻接矩阵形式

时间复杂度为 $O(n^2)$，在涉及[边的增删](https://ac.nowcoder.com/acm/contest/view-submission?submissionId=47959247)（虚）时比较方便。

~~~c++
namespace Dijkstra {
    const int MAXN = 1010;
    bool vis[MAXN];
    int pre[MAXN]; //pre[v] = u 表示v的前驱节点为u
    deque<pair<int, int>> road; //最短路径
    int g[MAXN][MAXN];
    ll dis[MAXN];
    void dij_init(int n) { /* 注意邻接矩阵的初始化 */
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++) 
                if (i == j)
                    g[i][j] = 0;
                else
                    g[i][j] = INF;
    }
    void dijkstra(int n, int start) {
        for (int i = 1; i <= n; i++) {
            dis[i] = INF, vis[i] = false, pre[i] = -1;
        }
        dis[start] = 0;
        for (int j = 1; j <= n; j++) {
            int u = -1;
            ll minn = INF;
            for (int i = 1; i <= n; i++) {
                if (!vis[i] && dis[i] < minn) {
                    minn = dis[i];
                    u = i;
                }
            }
            if (u == -1)
                break;
            vis[u] = true;
            for (int v = 1; v <= n; v++) {
                if (!vis[v] && dis[u] + g[u][v] < dis[v]) {
                    dis[v] = dis[u] + g[u][v];
                    pre[v] = u;
                }
            }
        }
    }
    void dij_getRoad(int end) { //传入终点，得到一条最短路径，存储在road中
        int tmp = end;
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

###### 有向图删边

~~~c++
for (ll i = 0, u, v, w; i < road.size(); i++) {
    u = road[i].first, v = road[i].second, w = g[u][v];
    g[u][v] = g[v][u] = INF; //删
    dijkstra(n, x);
    ll dis1 = dis[s] + dis[t];
    dijkstra(n, s);
    ll dis2 = dis[t];
    if (dis1 == dis2) {
        ckmin(ans, id[u][v]);
    }
    g[u][v] = g[v][u] = w; //返回原状态
}
~~~

#### 堆优化 Dijkstra

时间复杂度：$O((n + m) \log m)$

##### 使用 vector

vector 的版本，在边数或点数过多的时候可能会 MLE ：

~~~c++
namespace Dijkstra {
    const int maxn = 1000010;
    struct qnode {
        int v;
        ll c;
        qnode(int _v = 0, ll _c = 0) : v(_v), c(_c) {}
        bool operator<(const qnode &t) const {
            return c > t.c;
        }
    };
    struct Edge {
        int v;
        ll cost;
        Edge(int _v = 0, ll _cost = 0) : v(_v), cost(_cost) {}
    };
    vector<Edge> g[maxn];
    bool vis[maxn];
    ll dis[maxn];
    //点的编号从1开始
    void dijkstra(int n, int start) {
        for (int i = 1; i <= n; i++) {
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
            int u = t.v;
            if (vis[u])
                continue;
            vis[u] = true;
            for (int i = 0; i < g[u].size(); i++) {
                int v = g[u][i].v;
                ll c = g[u][i].cost;
                if (!vis[v] && dis[v] > dis[u] + c) {
                    dis[v] = dis[u] + c;
                    q.push({v, dis[v]});
                }
            }
        }
    }
    void addedge(int u, int v, ll w) {
        g[u].push_back({v, w});
    }
}
using namespace Dijkstra;
~~~

##### 使用链式前向星

~~~c++
namespace Dijkstra {
    const int maxn = 1000010;
    struct qnode {
        int v;
        ll c;
        qnode(int _v = 0, ll _c = 0) : v(_v), c(_c) {}
        bool operator<(const qnode &t) const {
            return c > t.c;
        }
    };
    struct Edge {
        int v, next;
        ll cost;
        Edge(int _v = 0, ll _cost = 0, int _next = 0) : v(_v), cost(_cost), next(_next) {}
    } g[maxn];
    // vector<Edge> g[maxn];
    int head[N];
    bool vis[N];
    ll dis[N];
    //点的编号从1开始
    int d_cnt = 0;
    void dijkstra(int n, int start) {
        for (int i = 1; i <= n; i++) {
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
            int u = t.v;
            if (vis[u])
                continue;
            vis[u] = true;
            for (int i = head[u]; ~i; i = g[i].next) {
                int v = g[i].v;
                ll c = g[i].cost;
                if (!vis[v] && dis[v] > dis[u] + c) {
                    dis[v] = dis[u] + c;
                    q.push({v, dis[v]});
                }
            }
        }
    }
    void addedge(int u, int v, ll w) {
        g[d_cnt].v = v;
        g[d_cnt].cost = w;
        g[d_cnt].next = head[u];
        head[u] = d_cnt++;
    }
    void init(int n) {
//        memset(head, -1, sizeof(head));
        for (int i = 0; i <= n; i++) {
            head[i] = -1;
        }
    }
}
using namespace Dijkstra;
~~~

#### SPFA

首先用数组 dis 记录起点到每个结点的最短路径，用邻接表来存储图，用 vis 数组记录当前节点是否在队列中。

具体操作为：用队列来保存待优化的结点（类似于 BFS ），优化时每次取出队首结点，并且用队首节点来对最短路径进行更新并进行松弛操作。如果要对所连点的最短路径需要更新，且该点不在当前的队列中，就将该点加入队列。然后不断进行松弛操作，直至队列空为止。

总结来说就是，只要某个点 u 的 dis[u] 得到更新，并且此时不在队列中，就将其入队，目的是为了以u为基点进一步更新它的邻接点 v 的 dis[v] 。

这个是队列实现，有时候改成栈实现会更加快。

可以处理负权边和判定负环回路。

时间复杂度：$O(kE)$

~~~c++
namespace SPFA {
    const int maxn = 1000010;
    struct Edge {
        int v;
        ll cost;
        Edge(int _v = 0, ll _cost = 0):v(_v), cost(_cost) {}
    };
    vector<Edge> g[maxn];
    void addedge(int u, int v, ll w) {
        g[u].push_back({v, w});
    }
    bool vis[maxn]; //在队列标志
    int cnt[maxn]; //每个点的入队列次数
    ll dis[maxn];
    bool spfa(int start, int n) {
        for (int i = 0; i <= n; i++) {
            vis[i] = false;
            dis[i] = INF;
            cnt[i] = 0;
        }
        vis[start] = true;
        dis[start] = 0;
        queue<int> q;
        while (!q.empty())
            q.pop();
        q.push(start);
        cnt[start] = 1;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            vis[u] = false;
            for (int i = 0; i < g[u].size(); i++) {
                int v = g[u][i].v;
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

#### Floyd

解决任意两点间的最短路径的一种算法，可以正确处理有向图或负权的最短路径问题，同时也被用于计算有向图的传递闭包。Floyd-Warshall 算法的时间复杂度为 $O(N^3)$ ，空间复杂度为 $O(N^2)$ 。

最外层循环相当于用k点作为中转点对全图进行更新。

~~~c++
namespace Floyd {
    constexpr int maxn = 1010;
    int n;
    ll dis[maxn][maxn];
    void Init() { //先初始化边权数组
        memset(dis, 0x3f, sizeof(dis));
        for (int i = 1; i <= n; i++) {
            dis[i][i] = 0;
        }
    }
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
using namespace Floyd;
~~~

### 最小生成树

#### Kruskal

~~~c++
int fa[N];
struct Edge {
    int u, v, w;
} e[N << 1];
int tot = 0;
void addedge(int u, int v, int w) {
    e[tot].u = u;
    e[tot].v = v;
    e[tot++].w = w;
}

int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }

//传入点数，返回最小生成树的权值，如果不连通返回-1
int kruskal(int n) {
    for (int i = 1; i <= n; i++)
        fa[i] = i;
    sort(e, e + tot, [](const Edge &A, const Edge &B) { return A.w < B.w; });
    int cnt = 0, ans = 0;
    for (int i = 0, u, v, w; i < tot; i++) {
        u = e[i].u, v = e[i].v, w = e[i].w;
        int t1 = find(u), t2 = find(v);
        if (t1 != t2)
            ans += w, fa[t1] = t2, ++cnt;
        if (cnt == n - 1)
            break;
    }
    if (cnt == n - 1) return ans;
    else return -1;
}
~~~

#### Prim

~~~c++
namespace Prim{
/*
 * 稠密图（|E|接近|V|^2）时使用Prim
 * Prim 求 MST
 * 耗费矩阵 cost[][]，标号从 0 开始，0∼n-1
 * 返回最小生成树的权值，返回 -1 表示原图不连通
 */
    const int MAXN = 5010;
    bool vis[MAXN];
    ll lowc[MAXN];
    //点是 [0, n-1]
    ll prim(ll c[][MAXN], int n) {
        ll ans = 0;
        memset(vis, 0, sizeof(vis));
        vis[0] = true;
        for (int i = 1; i < n; i++) 
            lowc[i] = c[0][i];
        for (int i = 1; i < n; i++) {
            ll minc = INF;
            int p = -1;
            for (int j = 0; j < n; j++)
                if (!vis[j] && minc > lowc[j])
                    minc = lowc[j], p = j;
            if (minc == INF) //原图不连通
                return -1;
            ans += minc;
            vis[p] = true;
            for (int j = 0; j < n; j++)
                if (!vis[j] && lowc[j] > c[p][j])
                    lowc[j] = c[p][j];
        }
        return ans;
    }
}
using namespace Prim;
~~~

### 最长路

[P1807 最长路 - 洛谷](https://www.luogu.com.cn/problem/P1807)

一个负数越小，那它的绝对值肯定更大。这样我们就可以把最长路问题转换为最短路问题。

如果使用 SPFA ，我们在存边时将边权取相反数，求最短路，答案取相反数即为所求的最长路。

如果使用拓扑排序，则先需要保证图为 DAG ，在拓扑的过程中更新每个点的最长路。

~~~c++
int ind[N];
bool isReachable[N]; //标记是否可以从1走到这个点
int dis[N];

//         起点    终点    点数
void Topo(int st, int en, int n) {
    isReachable[st] = true;
    dis[en] = -1; //-1表示不可达

    queue<int> q;
    for (int i = 1; i <= n; i++)
        if (ind[i] == 0)
            q.push(i);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = head[u]; ~i; i = g[i].next) {
            int v = g[i].v, w = g[i].w;
            ind[v]--;
            if (isReachable[u]) { //如果能从1走到u，则可以通过当前这条边走到v
                ckmax(dis[v], dis[u] + w);
                isReachable[v] = true;
            }
            if (ind[v] == 0) {
                q.push(v);
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n, m; cin >> n >> m;
    for (int i = 0; i <= n; i++) {
        head[i] = -1;
    }
    for (int i = 0, u, v, w; i < m; i++) {
        cin >> u >> v >> w;
        addedge(u, v, w);
        ind[v]++;
    }
    Topo(1, n, n);
    cout << dis[n] << "\n";
    return 0;
}
~~~

### 最小环

#### 问题

给出一个图，问其中的有 $n$ 个节点构成的边权和最小的环 $n \ge 3$ 是多大。图的最小环也称**围长**。

#### 暴力解法

设 $u, v$ 之间有一条长为 $w$ 的边，$dis(u, v)$ 表示删除 $u$ 和 $v$ 之间的连边之后，$u$ 和 $v$ 的最短路。那么最小环是 $dis(u, v) + w$。时间复杂度为 $\mathcal O(n^2m)$

#### Dijkstra

如果枚举所有边，每一次求删除一条边之后对这条边的起点跑一次 Dijkstra，时间复杂度为：$\mathcal O(m(n + m) \log m)$

如果是像[CCPC2021桂林 E - Buy and Delete](https://codeforces.com/gym/103409/problem/E)一样，最小环可以是 $n$ 个节点构成的边权和最小的环 $n \ge 2$，那么我们可以直接对每个点进行一次 Dijkstra，然后 $\mathcal O(n^2)$ 枚举可能的环起点和中断点（就是把环分成两部分），更新围长。

#### Floyd

时间复杂度：$\mathcal O(n^3)$。

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
template <class T> bool ckmin(T &a, const T &b) { return b < a ? a = b, 1 : 0; }

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

### 二分图

dfs版本的匈牙利算法，时间复杂度为$O(VE)$。

一些定义和定理：

**最大匹配数**：最大匹配的匹配边的数目

**最小点覆盖数**：选取最少的点，使任意一条边至少有一个端点被选择

**最大独立数**：选取最多的点，使任意所选两点均不相连

**最小路径覆盖数**：对于一个 DAG（有向无环图），选取最少条路径，使得每个顶点属于且仅属于一条路径。路径长可以为 0（即单个点）。

定理1：最大匹配数 = 最小点覆盖数（这是 Konig 定理）

定理2：最大匹配数 = 最大独立数

定理3：最小路径覆盖数 = 顶点数 - 最大匹配数

~~~c++
vector<int> G[N << 1];
bool vis[N << 1];
int match[N << 1];

bool dfs(int u) {
    for (auto &v : G[u]) {
        if (vis[v])
            continue;
        vis[v] = 1;
        if (match[v] == -1 || dfs(match[v])) {  // 如果v没有匹配，或者v的匹配找到了新的匹配
            match[v] = u; // 更新匹配信息
            return 1;
        }
    }
    return 0;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n, m, e; cin >> n >> m >> e;
    for (int i = 0, u, v; i < e; i++) {
        cin >> u >> v;
        G[u].emplace_back(v + n);
        G[v + n].emplace_back(u);
    }
    int mtc = 0;
    memset(match, -1, sizeof(match));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= 2 * n; j++)
            vis[j] = false;
        if (dfs(i))
            ++mtc;
    }
    cout << mtc << "\n";
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

### 费用流

#### 类 dinic 

时间复杂度为 $O(f \left | V \right | \left | E \right |)$，$f$ 为最大流量。

~~~c++
namespace MCMF {
    const int MAXN = 5007, MAXM = 100010, INF = 0x3f3f3f3f;
    int head[MAXN], cnt = 1;
    struct Edge
    {
        int to, w, c, next;
    } edges[MAXM * 2];
    inline void add(int from, int to, int w, int c)
    {
        edges[++cnt] = {to, w, c, head[from]};
        head[from] = cnt;
    }
    inline void addEdge(int from, int to, int w, int c)
    {
        add(from, to, w, c);
        add(to, from, 0, -c);
    }
    int s, t, dis[MAXN], cur[MAXN];
    bool inq[MAXN], vis[MAXN];
    queue<int> Q;
    bool SPFA()
    {
        while (!Q.empty())
            Q.pop();
        copy(head, head + MAXN, cur);
        fill(dis, dis + MAXN, INF);
        dis[s] = 0;
        Q.push(s);
        while (!Q.empty())
        {
            int p = Q.front();
            Q.pop();
            inq[p] = 0;
            for (int e = head[p]; e != 0; e = edges[e].next)
            {
                int to = edges[e].to, vol = edges[e].w;
                if (vol > 0 && dis[to] > dis[p] + edges[e].c)
                {
                    dis[to] = dis[p] + edges[e].c;
                    if (!inq[to])
                    {
                        Q.push(to);
                        inq[to] = 1;
                    }
                }
            }
        }
        return dis[t] != INF;
    }
    int dfs(int p = s, int flow = INF)
    {
        if (p == t)
            return flow;
        vis[p] = 1;
        int rmn = flow;
        for (int eg = cur[p]; eg && rmn; eg = edges[eg].next)
        {
            cur[p] = eg;
            int to = edges[eg].to, vol = edges[eg].w;
            if (vol > 0 && !vis[to] && dis[to] == dis[p] + edges[eg].c)
            {
                int c = dfs(to, min(vol, rmn));
                rmn -= c;
                edges[eg].w -= c;
                edges[eg ^ 1].w += c;
            }
        }
        vis[p] = 0;
        return flow - rmn;
    }
    int maxflow, mincost;
    inline void run(int s, int t)
    {
        MCMF::s = s, MCMF::t = t;
        while (SPFA())
        {
            int flow = dfs();
            maxflow += flow;
            mincost += dis[t] * flow;
        }
    }
} // namespace MCMF
~~~

#### Push-relabel 算法

时间复杂度$O(V^3)$？？？

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
constexpr const ll LL_INF = 0x3f3f3f3f3f3f3f3f;

template <const int MAXV, class flowUnit, class costUnit, const int SCALE = 8>
struct PushRelabelMinCostMaxFlow
{
    struct Edge
    {
        int to;
        flowUnit cap, resCap;
        costUnit cost;
        int rev;
        Edge(int to, flowUnit cap, costUnit cost, int rev) : to(to), cap(cap), resCap(cap), cost(cost), rev(rev) {}
    };
    int cnt[MAXV * 2], h[MAXV], stk[MAXV], top;
    flowUnit FLOW_EPS, maxFlow, ex[MAXV];
    costUnit COST_INF, COST_EPS, phi[MAXV], bnd, minCost, negCost;
    vector<int> hs[MAXV * 2];
    vector<Edge> adj[MAXV];
    typename vector<Edge>::iterator cur[MAXV];
    PushRelabelMinCostMaxFlow(flowUnit FLOW_EPS, costUnit COST_INF, costUnit COST_EPS) : FLOW_EPS(FLOW_EPS), COST_INF(COST_INF), COST_EPS(COST_EPS) {}
    void addEdge(int v, int w, flowUnit flow, costUnit cost)
    {
        if (v == w)
        {
            if (cost < 0)
                negCost += flow * cost;

            return;
        }

        adj[v].emplace_back(w, flow, cost, int(adj[w].size()));
        adj[w].emplace_back(v, 0, -cost, int(adj[v].size()) - 1);
    }
    void init(int V)
    {
        negCost = 0;

        for (int i = 0; i < V; i++)
            adj[i].clear();
    }
    flowUnit getMaxFlow(int V, int s, int t)
    {
        auto push = [&](int v, Edge &e, flowUnit df)
        {
            int w = e.to;

            if (abs(ex[w]) <= FLOW_EPS && df > FLOW_EPS)
                hs[h[w]].push_back(w);

            e.resCap -= df;
            adj[w][e.rev].resCap += df;
            ex[v] -= df;
            ex[w] += df;
        };

        if (s == t)
            return maxFlow = 0;

        fill(h, h + V, 0);
        h[s] = V;
        fill(ex, ex + V, 0);
        ex[t] = 1;
        fill(cnt, cnt + V * 2, 0);
        cnt[0] = V - 1;

        for (int v = 0; v < V; v++)
            cur[v] = adj[v].begin();

        for (int i = 0; i < V * 2; i++)
            hs[i].clear();

        for (auto &&e : adj[s])
            push(s, e, e.resCap);

        if (!hs[0].empty())
            for (int hi = 0; hi >= 0;)
            {
                int v = hs[hi].back();
                hs[hi].pop_back();

                while (ex[v] > FLOW_EPS)
                {
                    if (cur[v] == adj[v].end())
                    {
                        h[v] = INT_MAX;

                        for (auto e = adj[v].begin(); e != adj[v].end(); e++)
                            if (e->resCap > FLOW_EPS && h[v] > h[e->to] + 1)
                            {
                                h[v] = h[e->to] + 1;
                                cur[v] = e;
                            }

                        cnt[h[v]]++;

                        if (--cnt[hi] == 0 && hi < V)
                            for (int i = 0; i < V; i++)
                                if (hi < h[i] && h[i] < V)
                                {
                                    cnt[h[i]]--;
                                    h[i] = V + 1;
                                }

                        hi = h[v];
                    }
                    else if (cur[v]->resCap > FLOW_EPS && h[v] == h[cur[v]->to] + 1)
                        push(v, *cur[v], min(ex[v], cur[v]->resCap));
                    else
                        cur[v]++;
                }

                while (hi >= 0 && hs[hi].empty())
                    hi--;
            }

        return maxFlow = -ex[s];
    }
    pair<flowUnit, costUnit> getMaxFlowMinCost(int V, int s = -1, int t = -1)
    {
        auto costP = [&](int v, const Edge &e)
        {
            return e.cost + phi[v] - phi[e.to];
        };
        auto push = [&](int v, Edge &e, flowUnit df, bool pushToStack)
        {
            if (e.resCap < df)
                df = e.resCap;

            int w = e.to;
            e.resCap -= df;
            adj[w][e.rev].resCap += df;
            ex[v] -= df;
            ex[w] += df;

            if (pushToStack && FLOW_EPS < ex[e.to] && ex[e.to] <= df + FLOW_EPS)
                stk[top++] = e.to;
        };
        auto relabel = [&](int v, costUnit delta)
        {
            phi[v] -= delta + bnd;
        };
        auto lookAhead = [&](int v)
        {
            if (abs(ex[v]) > FLOW_EPS)
                return false;

            costUnit delta = COST_INF;

            for (auto &&e : adj[v])
            {
                if (e.resCap <= FLOW_EPS)
                    continue;

                costUnit c = costP(v, e);

                if (c < -COST_EPS)
                    return false;
                else
                    delta = min(delta, c);
            }

            relabel(v, delta);
            return true;
        };
        auto discharge = [&](int v)
        {
            costUnit delta = COST_INF;

            for (int i = 0; i < int(adj[v].size()); i++)
            {
                Edge &e = adj[v][i];

                if (e.resCap <= FLOW_EPS)
                    continue;

                if (costP(v, e) < -COST_EPS)
                {
                    if (lookAhead(e.to))
                    {
                        i--;
                        continue;
                    }

                    push(v, e, ex[v], true);

                    if (abs(ex[v]) <= FLOW_EPS)
                        return;
                }
                else
                    delta = min(delta, costP(v, e));
            }

            relabel(v, delta);
            stk[top++] = v;
        };
        minCost = 0;
        bnd = 0;
        costUnit mul = 2 << __lg(V);

        for (int v = 0; v < V; v++)
            for (auto &&e : adj[v])
            {
                minCost += e.cost * e.resCap;
                e.cost *= mul;
                bnd = max(bnd, e.cost);
            }

        maxFlow = (s == -1 || t == -1) ? 0 : getMaxFlow(V, s, t);
        fill(phi, phi + V, 0);
        fill(ex, ex + V, 0);

        while (bnd > 1)
        {
            bnd = max(bnd / SCALE, costUnit(1));
            top = 0;

            for (int v = 0; v < V; v++)
                for (auto &&e : adj[v])
                    if (costP(v, e) < -COST_EPS && e.resCap > FLOW_EPS)
                        push(v, e, e.resCap, false);

            for (int v = 0; v < V; v++)
                if (ex[v] > FLOW_EPS)
                    stk[top++] = v;

            while (top > 0)
                discharge(stk[--top]);
        }

        for (int v = 0; v < V; v++)
            for (auto &&e : adj[v])
            {
                e.cost /= mul;
                minCost -= e.cost * e.resCap;
            }

        return make_pair(maxFlow, (minCost /= 2) += negCost);
    }
};

const int MAXN = 405, MAXM = 15005;
PushRelabelMinCostMaxFlow<MAXN, ll, ll> mcmf(0, LL_INF, 0);


int main()
{
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    int n, m; cin >> n >> m;
    mcmf.init(n);
    for (int i = 0, s, t, c, w; i < m; i++) {
        cin >> s >> t >> c >> w;
        --s, --t;
        mcmf.addEdge(s, t, c, w);
    }
    pair<int, int> ans = mcmf.getMaxFlowMinCost(n, 0, n - 1);
    cout << ans.first << " " << ans.second << "\n";
    return 0;
}
~~~

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

### 数位 dp

数位 dp 主要解决与数位有关的问题。

其基本的思想就是记忆化搜索保存搜索过的状态，通过从高位到低位暴力枚举能在每个数位上出现的数，搜索出某一区间 $[L, R]$ 内的数，在搜索的过程中更新要求的答案。

比如求 $[L, R]$ 区间内满足某种性质的数的数量，即计数问题时，我们先利用前缀和思想转化为求 $[1, L - 1]$，$[1, R]$ 两个区间的问题，然后将数字按数位拆分出来，进行数位 dp 。

[ZJOI2010数字计数](https://www.luogu.com.cn/problem/P2602)

题意：给定两个正整数 $a$ 和 $b$，求在 $[a, b]$ 中的所有整数中，每个数码(digit)各出现了多少次。

limit 表示前面是否都贴着放。

lead 表示前面是否都是前导 0 。

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

值得注意的是，我们在利用前缀和性质的时候，留心判断lf - 1是否已经超出了数据范围，比如[P4124 [CQOI2016]手机号码](https://www.luogu.com.cn/problem/P4124)。

## 计算几何

### 凸包

使用 Graham 算法计算凸包的[周长](https://www.luogu.com.cn/problem/P2742)和[面积](https://vjudge.net/problem/POJ-3348)。

时间复杂度：$O(n \log n)$，$n$ 为平面上的点的数量。

~~~c++
struct Point {
    double x, y;
} p[N], st[N];
int n;

double check(Point a1, Point a2, Point b1, Point b2) {
    return (a2.x - a1.x) * (b2.y - b1.y) - (b2.x - b1.x) * (a2.y - a1.y);
}

double dis(Point p1, Point p2) {
    return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
}

bool cmp(Point p1, Point p2) {
    double res = check(p[1], p1, p[1], p2); //读入数据时保证p[1]是y值最小点
    if (res > 0 || res == 0 && dis(p[0], p1) < dis(p[0], p2))
        return 1; //极角排序
    return 0;
}

double getC(int cnt) { //cnt为凸包点的个数，返回凸包周长
    double C = 0.0;
    for (int i = 1; i <= cnt; i++)
        C += dis(st[i], st[i + 1]); //st[1, cnt]中存着凸包序列，将两两距离累加得到凸包周长
    return C;
}

double getS(Point a, Point b, Point c) //返回三角形面积
{
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2;
}

double getPS(int cnt) //cnt为凸包点的个数，返回多边形面积。必须确保 n>=3，且多边形是凸多边形
{
    double sumS = 0;
    for (int i = 2; i <= cnt - 1; i++)
        sumS += fabs(getS(st[1], st[i], st[i + 1]));
    return sumS;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> p[i].x >> p[i].y;
        if (i != 1 && p[i].y < p[1].y) { //保证p[1]是y值最小点
            double tmpx = p[1].x, tmpy = p[1].y;
            p[1].x = p[i].x, p[1].y = p[i].y;
            p[i].x = tmpx, p[i].y = tmpy;
        }
    }
    sort(p + 2, p + 1 + n, cmp);
    int cnt = 1; //凸包点的个数
    st[1] = p[1]; //最低点一定在凸包内
    for (int i = 2; i <= n; i++) {
        while (cnt > 1 && check(st[cnt - 1], st[cnt], st[cnt], p[i]) <= 0)
            cnt--;
        cnt++;
        st[cnt] = p[i];
    }
    st[cnt + 1] = p[1]; //最后一点回到凸包起点

    return 0;
}
~~~

### 平面最近点对

[参考](https://www.luogu.com.cn/blog/syksykCCC/solution-p1429)

~~~c++
struct Point {
    double x, y;
}p[N];
int n, tmp[N];
const double inf = 2e50;

double dis(int i, int j) {
    return sqrt(1ll * (p[i].x - p[j].x) * (p[i].x - p[j].x) + 1ll * (p[i].y - p[j].y) * (p[i].y - p[j].y));
}

double merge(int lf, int rt) { //返回由编号lf~rt的点构成的最近点对的距离
    double d = inf;
    if (lf == rt) //一个点，无效值
        return inf;
    if (lf + 1 == rt) 
        return dis(lf, rt);
    ll mid = lf + (rt - lf) / 2;
    double d1 = merge(lf, mid); //分治
    double d2 = merge(mid + 1, rt);
    d = min(d1, d2);
    int cnt = 0;
    for (int i = lf; i <= rt; i++) { //找跨越中界线的最近点对
        if (fabs(p[i].x - p[mid].x) < d) { //如果该点到中界线的距离已经>=d，更不用考虑另一侧的距离
            tmp[++cnt] = i;
        }
    }
    sort(tmp + 1, tmp + 1 + cnt, [](const int &i, const int &j) { return p[i].y < p[j].y; });
    for (int i = 1; i <= cnt; i++) {
        for (int j = i + 1; j <= cnt && p[tmp[j]].y - p[tmp[i]].y < d; j++) { //如果两点的纵坐标之差已经>=d，更不用考虑水平距离了
            double d3 = dis(tmp[i], tmp[j]);
            ckmin(d, d3);
        }
    }
    return d;
}

void solve() {
    cin >> n; //点数
    for (int i = 1; i <= n; i++) {
        cin >> p[i].x >> p[i].y;
    }
    sort(p + 1, p + 1 + n, [](const Point &a, const Point &b) { return a.x == b.x ? a.y < b.y : a.x < b.x; });
    cout << fixed << setprecision(4) << merge(1, n) << "\n";
}
~~~

## 搜索

### 二分图染色

对于任意一个连通图，每个点可以被染成 0 或 1 两种颜色。

如果存在一种染色方案，使被任意一条边连接的两个点 (u , v) 颜色不同，那么这个图就是一个二分图。

对于一个联通的无向图来说，如果它是一个二分图，那么其染色方式必定只有 2 种（因为对于一个点 n，其染色方式有且仅有 2 种）。

如果不存在一个二分图，那么当你重复上面的染色过程是，必定会遇到矛盾，判定矛盾的方式也很简单，如果存在一条边 (u , v )，使得当你搜到 v 时 u 已经被染成了与 v 相同的颜色，那么该图不为二分图，直接退出循环。

~~~c++
int d_cnt = 0; //点标号从0开始
struct Edge {
    int v, w;
    Edge(int v, int w): v(v), w(w) {}
};
vector<Edge> g[N];
 
int cnt[2]; //分别记录染成0和1的个数
int col[N];
 
void dfs(int u, int co, int d) { //二分图染色，d表示现在是原数的第d位上
    col[u] = co;
    if (co == 0) {
        cnt[0]++;
    } else {
        cnt[1]++;
    }
    for (int i = 0; i < (int)g[u].size(); i++) {
        int v = g[u][i].v, w = g[u][i].w;
        if (col[v] != -1) {
            if (col[v] != (col[u] ^ (w >> d & 1))) {
                cout << -1 << "\n";
                exit(0);
            }
        } else {
            dfs(v, col[u] ^ (w >> d & 1), d);
        }
    }
}
~~~

### IDA*

迭代加深：在DFS的搜索里面，可能会面临一个答案的层数很低，但是DFS搜到了另为一个层数很深的分支里面导致时间很慢，但是又卡BFS的空间，这个时候，可以限制DFS的搜索层数，一次又一次放宽搜索的层数，直到搜索到答案就退出，时间可观，结合了BFS和DFS的优点于一身。

定义函数：
$$
F ( x ) = G ( x ) + H ( x )
$$
其中G(x)表示已经走过的路，H(x)表示期望走的路。

IDA\*就是将F(x)搬到了IDDFS（迭代加深的DFS）上然后加了个可行性剪枝。

估值函数的确定：

1. 估值函数一般是当状态离目标越近时越优，当然是总体趋势，存在个别的。
2. 估值函数里的参数一般是比较明显的，且每次操作后一般会改变的，一般也会满足第一点。
3. 参数一般变化是有范围的。

如果你想要查找地图上任意一个位置到另一个位置的路径，那么请使用广度优先搜索（BFS）或 Dijkstra 算法。如果移动成本相同，则使用广度优先搜索；如果移动成本不同，则使用 Dijkstra 算法。

如果你要查找从一个位置到另一个位置，或者从一个位置到多个目标位置中最近的位置，请使用贪心最佳优先搜索算法或A\*。在大多数情况下建议使用A\*。当你使用贪心最佳优先算法时，请考虑使用带有“不可接受的”启发式算法的A\*。

> 可接受的启发式
> 在计算机科学中，特别是在与寻路相关的算法中，如果启发式函数从未过高估计达到目标的成本，即它估计从当前位置到达目标位置的成本不高于当前的可能的最低成本，则认为该函数是可接受的。

算法给出的结果是否为最佳路径？对于给定的图，广度优先搜索和 Dijkstra 算法能够保证找到最短路径，但贪心最佳优先算法不保证，**对于 A\* 来说，如果启发式算法给出的成本永远不大于真实的距离，那么它也是可以保证给出最短路径**。随着启发式算法给出的成本越来越小，A\* 退化成 Dijkstra 算法；随着启发式算法给出的成本越来越高，A\* 退化成贪心最佳优先算法。

~~~c++
//P1379 八数码难题
string st, en = "123804765";

ll depth = 0; //搜索深度

//估价函数
ll calH(string s) {
    ll res = 0;
    for (ll i = 0; i < s.size(); i++) { //每个数字的曼哈顿距离之和
        if (s[i] == '0')
            continue;
        int p = en.find(s[i]);
        int enx = p / 3, eny = p % 3;
        int sx = i / 3, sy = i % 3;
        res += abs(sx - enx) + abs(sy - eny);
    }
    return res;
}

//now为当前步数，pre为上一步的位置，depth为搜索深度
bool IDA_star(ll now, ll pre) {
    ll est = calH(st);
    if (est == 0)
        return true;
    if (now + est > depth) //当前步数+估计值>深度限制，立即回溯
        return false;
    int pos = st.find('0'), x = pos / 3, y = pos % 3;
    for (int i = 0; i < 4; i++) {
        int dx = x + dir[i][0], dy = y + dir[i][1];
        if (dx < 0 || dx > 2 || dy < 0 || dy > 2 || dx * 3 + dy == pre/* 不走回头路 */)
            continue;
        swap(st[pos], st[dx * 3 + dy]);
        if (IDA_star(now + 1, pos))
            return true;
        swap(st[pos], st[dx * 3 + dy]);
    }
    return false;
}

int main() {
    cin >> st;
    while (!IDA_star(0, -1))
        ++depth;
    cout << depth << "\n";
    return 0;
}
~~~



## 其他

### 快读

~~~c++
inline char nc() {
    static char buf[1 << 24], *p1 = buf, *p2 = buf;
    return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 24, stdin), p1 == p2) ? EOF : *p1++;
}

template <class T>
inline void read(T &sum) {
    char ch = nc(); int tf = 0; sum = 0;
    while ((ch < '0' || ch > '9') && (ch != '-')) 
        ch = nc();
    tf = ((ch == '-') && (ch = nc()));
    while (ch >= '0' && ch <= '9')
        sum = sum * 10 + (ch - 48), ch = nc();
    (tf) && (sum = -sum);
}
template <class A, class... B> void read(A &a, B &...b) { read(a); read(b...); }

template <class T>
inline void print(T x) {
    if (x < 0)
        x = ~x + 1, putchar('-');
    if (x > 9)
        print(x / 10);
    putchar(x % 10 + '0');
}
~~~

### __int128

int128整数的输入输出

~~~c++
inline __int128 read() {
    __int128 x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-')
            f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}

inline void write(__int128 x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x > 9)
        write(x / 10);
    putchar(x % 10 + '0');
}

int main() {
    __int128 a = read();
    __int128 b = read();
    write(a + b);
    return 0;
}
~~~

### c++ 高精度模板

~~~c++
const int Base = 1000000000;
const int Capacity = 500;

struct BigInt
{
    int Len;
    int Data[Capacity];
    BigInt() : Len(0) {}
    BigInt(const BigInt &V) : Len(V.Len) { memcpy(Data, V.Data, Len * sizeof *Data); }
    BigInt(int V) : Len(0)
    {
        for (; V > 0; V /= Base)
            Data[Len++] = V % Base;
    }
    BigInt &operator=(const BigInt &V)
    {
        Len = V.Len;
        memcpy(Data, V.Data, Len * sizeof *Data);
        return *this;
    }
    int &operator[](int Index) { return Data[Index]; }
    int operator[](int Index) const { return Data[Index]; }
};

int compare(const BigInt &A, const BigInt &B)
{
    if (A.Len != B.Len)
        return A.Len > B.Len ? 1 : -1;
    int i;
    for (i = A.Len - 1; i >= 0 && A[i] == B[i]; i--)
        ;
    if (i < 0)
        return 0;
    return A[i] > B[i] ? 1 : -1;
}

BigInt operator+(const BigInt &A, const BigInt &B)
{
    int i, Carry(0);
    BigInt R;
    for (i = 0; i < A.Len || i < B.Len || Carry > 0; i++)
    {
        if (i < A.Len)
            Carry += A[i];
        if (i < B.Len)
            Carry += B[i];
        R[i] = Carry % Base;
        Carry /= Base;
    }
    R.Len = i;
    return R;
}

BigInt operator-(const BigInt &A, const BigInt &B)
{
    int i, Carry(0);
    BigInt R;
    R.Len = A.Len;
    for (i = 0; i < R.Len; i++)
    {
        R[i] = A[i] - Carry;
        if (i < B.Len)
            R[i] -= B[i];
        if (R[i] < 0)
            Carry = 1, R[i] += Base;
        else
            Carry = 0;
    }
    while (R.Len > 0 && R[R.Len - 1] == 0)
        R.Len--;
    return R;
}

BigInt operator*(const BigInt &A, const int &B)
{
    int i;
    ll Carry(0);
    BigInt R;
    for (i = 0; i < A.Len || Carry > 0; i++)
    {
        if (i < A.Len)
            Carry += ll(A[i]) * B;
        R[i] = Carry % Base;
        Carry /= Base;
    }
    R.Len = i;
    return R;
}

istream &operator>>(istream &In, BigInt &V)
{
    char Ch;
    for (V = 0; In >> Ch;)
    {
        V = V * 10 + (Ch - '0');
        if (In.peek() <= ' ')
            break;
    }
    return In;
}

ostream &operator<<(ostream &Out, const BigInt &V)
{
    int i;
    Out << (V.Len == 0 ? 0 : V[V.Len - 1]);
    for (i = V.Len - 2; i >= 0; i--)
        for (int j = Base / 10; j > 0; j /= 10)
            Out << V[i] / j % 10;
    return Out;
}

int main()
{
    vector<BigInt> f(10010, 0);
    BigInt a, b;
    cin >> a >> b;
    cout << a + b << endl;
    if (compare(a, b) > 0)
        cout << a - b << endl;
    else
        cout << b - a << endl;
    cout << a * 328478 << " " << b * 982347283 << endl;
}
~~~



### 模拟退火

模拟退火算法是通过赋予搜索过程一种时变且最终趋于零的概率突跳性，从而可有效避免陷入局部极小并最终趋于全局最优的串行结构的优化算法。

算法从某一较高初温出发，伴随温度参数的不断下降,结合一定的概率突跳特性在解空间中随机寻找目标函数的全局最优解，即在局部最优解能概率性地跳出并最终趋于全局最优。

使用时重写f函数和随机生成的点Node即可，注意多次调用SA来减小误差。如果求全局最小值则$de < 0$时接受当前解，如果求全局最大值则$de>0$时接受当前解。

~~~c++
double ans = 1e18;
const double delta = 0.993;
int n;
struct Node {
    double x, y, z;
} pnow, pans, p[N];
 
inline double sqr(double x) { return x * x; }
 
double f(Node now) {
    double res = 0;
    for (int i = 0; i < n; i++) {
        ckmax(res, sqrt(sqr(now.x - p[i].x) + sqr(now.y - p[i].y) + sqr(now.z - p[i].z)));
    }
    return res;
}
 
void SA() {
    pnow = pans; //初始化
    double T = 2021; //初始温度
    while (T > 1e-14) {
        Node ptmp = {pnow.x + ((rand() << 1) - RAND_MAX) * T, pnow.y + ((rand() << 1) - RAND_MAX) * T, pnow.z + ((rand() << 1) - RAND_MAX) * T}; //球心转移
        double new_ans = f(ptmp); //计算新解
        double de = new_ans - ans; //计算新解与当前最优解的插值
        if (de < 0) { //新解小于当前解则接受
            pans = ptmp;
            pnow = ptmp;
            ans = new_ans;
        } else if (exp(-de / T) * RAND_MAX > rand()) { //否则以概率接受
            pnow = ptmp;
        }
        T *= delta; //降温
    }
}
 
int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n;
    for (int i = 0; i < n; i++)
        cin >> p[i].x >> p[i].y >> p[i].z;
    SA(); //多次运行减小误差
    SA();
    SA();
    SA();
    SA();
    cout << fixed << setprecision(15) << ans << "\n";
    return 0;
}
~~~

### 三分法

#### 三分套三分

比如求[最小球覆盖](https://vjudge.net/problem/Gym-101981D)。显然题目所求的这个半径是一个关于 $x, y, z$ 的函数，即$f(x, y, z) = r$，我们考虑三分套三分求解这个函数的最小值。对于这个函数，无论固定  $x$，固定 $y$，还是固定 $z$，都易证它是一个单峰函数。所以可以先三分一个变量，再固定这个变量，三分另一个变量，来求出最后的答案。

~~~c++
ll n;
struct Node {
    ll x, y, z;
} p[N];
double cd[3];

inline double sqr(double x) { return x * x; }

double dis() {
    double ans = 0;
    for (ll i = 1; i <= n; i++) {
        ckmax(ans, sqrt(sqr(cd[0] - p[i].x) + sqr(cd[1] - p[i].y) + sqr(cd[2] - p[i].z)));
    }
    return ans;
}

double d(ll cnt) {
    if (cnt == 3)
        return dis();
    double l = -1e5, r = 1e5, mid1, mid2, f1, f2, ans = 1e18;
    while (l + eps < r) {
        mid1 = l + (r - l) / 3, mid2 = r - (r - l) / 3;
        cd[cnt] = mid1;
        f1 = d(cnt + 1);
        cd[cnt] = mid2;
        f2 = d(cnt + 1);
        if (f1 < f2) r = mid2, ans = f1;
        else l = mid1, ans = f2;
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> n;
    for (ll i = 1; i <= n; i++) {
        cin >> p[i].x >> p[i].y >> p[i].z;
    }
    cout << fixed << setprecision(8) << d(0) << "\n";
    return 0;
}
~~~

注意其递归的方式：

~~~
cnt==1: x		  0                   1
                /   \               /   \
cnt==2: y      0     1             0     1
              / \   / \           / \   / \
cnt==3: z    0   1 0   1         0   1 0   1
~~~

这样可以枚举出所有的排列。

#### 整数三分

注意控制左右边界跳出循环的条件，可以是设定一个较为宽松的左右边界，然后暴力枚举左右边界内的数值。

还是一个三分套三分的例子

[Submission #136129374 - Codeforces](https://codeforces.com/contest/1486/submission/136129374)

~~~c++
ll sol(vector<pair<ll, ll>> &p, vector<ll> cp, int step) {
    if (step == 2) {
        ll sum = 0;
        for (int i = 0; i < (int)p.size(); i++) {
            sum += dis(cp, p[i]);
        }
        if (sum < mind) {
            mind = sum;
            center = cp;
        }
        return sum;
    }
    ll l = 0, r = 1000000000, tans = LLONG_MAX;
    while (r - l >= 10) {
        ll m1 = (2ll * l + r) / 3, m2 = (l + 2ll * r) / 3;
        //ll m1 = l + (r - l + 1) / 3, m2 = r - (r - l + 1) / 3;
        cp[step] = m1;
        ll f1 = sol(p, cp, step + 1);
        cp[step] = m2;
        ll f2 = sol(p, cp, step + 1);
        if (f1 <= f2) {
            r = m2 - 1;
        } else {
            l = m1 + 1;
        }
    }
    for (ll tm = l; tm <= r; tm++) {
        cp[step] = tm;
        ll tf = sol(p, cp, step + 1);
        if (tf < tans) {
            tans = tf;
        }
    }
    return tans;
}
~~~

### Modint

tourist 的 modint [模板](https://atcoder.jp/contests/abc212/submissions/24652558)，注意模数的选择。

~~~c++
template <typename T>
T inverse(T a, T m)
{
    T u = 0, v = 1;
    while (a != 0)
    {
        T t = m / a;
        m -= t * a;
        swap(a, m);
        u -= t * v;
        swap(u, v);
    }
    assert(m == 1);
    return u;
}

template <typename T>
class Modular
{
public:
    using Type = typename decay<decltype(T::value)>::type;

    constexpr Modular() : value() {}
    template <typename U>
    Modular(const U &x)
    {
        value = normalize(x);
    }

    template <typename U>
    static Type normalize(const U &x)
    {
        Type v;
        if (-mod() <= x && x < mod())
            v = static_cast<Type>(x);
        else
            v = static_cast<Type>(x % mod());
        if (v < 0)
            v += mod();
        return v;
    }

    const Type &operator()() const { return value; }
    template <typename U>
    explicit operator U() const { return static_cast<U>(value); }
    constexpr static Type mod() { return T::value; }

    Modular &operator+=(const Modular &other)
    {
        if ((value += other.value) >= mod())
            value -= mod();
        return *this;
    }
    Modular &operator-=(const Modular &other)
    {
        if ((value -= other.value) < 0)
            value += mod();
        return *this;
    }
    template <typename U>
    Modular &operator+=(const U &other) { return *this += Modular(other); }
    template <typename U>
    Modular &operator-=(const U &other) { return *this -= Modular(other); }
    Modular &operator++() { return *this += 1; }
    Modular &operator--() { return *this -= 1; }
    Modular operator++(int)
    {
        Modular result(*this);
        *this += 1;
        return result;
    }
    Modular operator--(int)
    {
        Modular result(*this);
        *this -= 1;
        return result;
    }
    Modular operator-() const { return Modular(-value); }

    template <typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, int>::value, Modular>::type &operator*=(const Modular &rhs)
    {
#ifdef _WIN32
        uint64_t x = static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value);
        uint32_t xh = static_cast<uint32_t>(x >> 32), xl = static_cast<uint32_t>(x), d, m;
        asm(
            "divl %4 \n\t"
            : "=a"(d), "=d"(m)
            : "d"(xh), "a"(xl), "r"(mod()));
        value = m;
#else
        value = normalize(static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value));
#endif
        return *this;
    }
    template <typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, long long>::value, Modular>::type &operator*=(const Modular &rhs)
    {
        long long q = static_cast<long long>(static_cast<long double>(value) * rhs.value / mod());
        value = normalize(value * rhs.value - q * mod());
        return *this;
    }
    template <typename U = T>
    typename enable_if<!is_integral<typename Modular<U>::Type>::value, Modular>::type &operator*=(const Modular &rhs)
    {
        value = normalize(value * rhs.value);
        return *this;
    }

    Modular &operator/=(const Modular &other) { return *this *= Modular(inverse(other.value, mod())); }

    friend const Type &abs(const Modular &x) { return x.value; }

    template <typename U>
    friend bool operator==(const Modular<U> &lhs, const Modular<U> &rhs);

    template <typename U>
    friend bool operator<(const Modular<U> &lhs, const Modular<U> &rhs);

    template <typename V, typename U>
    friend V &operator>>(V &stream, Modular<U> &number);

private:
    Type value;
};

template <typename T>
bool operator==(const Modular<T> &lhs, const Modular<T> &rhs) { return lhs.value == rhs.value; }
template <typename T, typename U>
bool operator==(const Modular<T> &lhs, U rhs) { return lhs == Modular<T>(rhs); }
template <typename T, typename U>
bool operator==(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) == rhs; }

template <typename T>
bool operator!=(const Modular<T> &lhs, const Modular<T> &rhs) { return !(lhs == rhs); }
template <typename T, typename U>
bool operator!=(const Modular<T> &lhs, U rhs) { return !(lhs == rhs); }
template <typename T, typename U>
bool operator!=(U lhs, const Modular<T> &rhs) { return !(lhs == rhs); }

template <typename T>
bool operator<(const Modular<T> &lhs, const Modular<T> &rhs) { return lhs.value < rhs.value; }

template <typename T>
Modular<T> operator+(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) += rhs; }
template <typename T, typename U>
Modular<T> operator+(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) += rhs; }
template <typename T, typename U>
Modular<T> operator+(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) += rhs; }

template <typename T>
Modular<T> operator-(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) -= rhs; }
template <typename T, typename U>
Modular<T> operator-(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) -= rhs; }
template <typename T, typename U>
Modular<T> operator-(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) -= rhs; }

template <typename T>
Modular<T> operator*(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) *= rhs; }
template <typename T, typename U>
Modular<T> operator*(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) *= rhs; }
template <typename T, typename U>
Modular<T> operator*(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) *= rhs; }

template <typename T>
Modular<T> operator/(const Modular<T> &lhs, const Modular<T> &rhs) { return Modular<T>(lhs) /= rhs; }
template <typename T, typename U>
Modular<T> operator/(const Modular<T> &lhs, U rhs) { return Modular<T>(lhs) /= rhs; }
template <typename T, typename U>
Modular<T> operator/(U lhs, const Modular<T> &rhs) { return Modular<T>(lhs) /= rhs; }

template <typename T, typename U>
Modular<T> power(const Modular<T> &a, const U &b)
{
    assert(b >= 0);
    Modular<T> x = a, res = 1;
    U p = b;
    while (p > 0)
    {
        if (p & 1)
            res *= x;
        x *= x;
        p >>= 1;
    }
    return res;
}

template <typename T>
bool IsZero(const Modular<T> &number)
{
    return number() == 0;
}

template <typename T>
string to_string(const Modular<T> &number)
{
    return to_string(number());
}

// U == std::ostream? but done this way because of fastoutput
template <typename U, typename T>
U &operator<<(U &stream, const Modular<T> &number)
{
    return stream << number();
}

// U == std::istream? but done this way because of fastinput
template <typename U, typename T>
U &operator>>(U &stream, Modular<T> &number)
{
    typename common_type<typename Modular<T>::Type, long long>::type x;
    stream >> x;
    number.value = Modular<T>::normalize(x);
    return stream;
}

using ModType = int;

/*
struct VarMod
{
    static ModType value;
};
ModType VarMod::value;
ModType &md = VarMod::value;
using Mint = Modular<VarMod>;
*/

constexpr int md = 1000000007; //998244353
using Mint = Modular<std::integral_constant<decay<decltype(md)>::type, md>>;

vector<Mint> fact(1, 1);
vector<Mint> inv_fact(1, 1);

/*Mint C(int n, int k)
{
    if (k < 0 || k > n)
    {
        return 0;
    }
    while ((int)fact.size() < n + 1)
    {
        fact.push_back(fact.back() * (int)fact.size());
        inv_fact.push_back(1 / fact.back());
    }
    return fact[n] * inv_fact[k] * inv_fact[n - k];
}*/
~~~

