# Nepenthe8's template

## 字符串处理

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

## 搜索

## 动态规划

## 计算几何

## 其他
