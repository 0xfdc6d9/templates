# Nepenthe8's template

## 字符串处理

### 字符串哈希

~~~c++
typedef unsigned long long ull;
ull base = 131;
ull mod = 212370440130137957ll;
constexpr int maxn = 10010;

ull hashs(char s[]) { //var = hashs(string)
    int len = strlen(s);
    ull ans = 0;
    for (int i = 0; i < len; i++)
        ans = (ans * base + (ull)s[i]) % mod;
    return ans;
}
~~~



## 数学

### 欧拉筛

~~~c++
constexpr int maxm = 10000010;
LL np[maxm], p[maxm], f[maxm], pn; //not prime(bool), prime[], f[i] is the smallest positive number m such that n/m is a square.

void Euler() {
	f[1] = 1;
    for (LL i = 2; i < maxm; i += 1) {
        if (not np[i]) {
            f[i] = i;
            p[pn ++] = i;
        }
        for (LL j = 0; j < pn; j += 1) {
            LL k = i * p[j];
            if (k >= maxm) break;
            np[k] = 1;
            if (f[i] % p[j]) f[k] = f[i] * p[j];
            else f[k] = f[i] / p[j];
            if (i % p[j] == 0) break;
        }
    }
}
~~~



## 数据机构

## 图论

### 最小环

[HDU-1599](https://vjudge.gxu.mmmm.mn/problem/HDU-1599)

~~~c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 210;
const ll INF = 0x3f3f3f3f;
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
