可以，建議把目前這套評分正式統整成下面這個版本。先**完全不考慮持股數、交易股數、資金量**，只判斷 Agent 的行為方向是否符合 Future Ranking。

# Overall Score 評分公式

## 1. Future Ranking 分數 (R_{t,i})

對第 (t) 天、第 (i) 檔股票，根據 Future Ranking 給分：

[
R_{t,i}=
\begin{cases}
+5 & \text{Top1}\
+4 & \text{Top2}\
+3 & \text{Top3}\
+2 & \text{Top4}\
+1 & \text{Top5}\
-1 & \text{Bottom5}\
-2 & \text{Bottom4}\
-3 & \text{Bottom3}\
-4 & \text{Bottom2}\
-5 & \text{Bottom1}\
0 & \text{Others}
\end{cases}
]

也就是：

| Future Ranking | (R) |
| -------------- | --: |
| Top1           |  +5 |
| Top2           |  +4 |
| Top3           |  +3 |
| Top4           |  +2 |
| Top5           |  +1 |
| Others         |   0 |
| Bottom5        |  -1 |
| Bottom4        |  -2 |
| Bottom3        |  -3 |
| Bottom2        |  -4 |
| Bottom1        |  -5 |

核心概念：

[
\text{未來越好的股票}\Rightarrow R越大
]

[
\text{未來越差的股票}\Rightarrow R越小
]

---

# 2. Agent 行為分數 (A_{t,i})

把 Agent 當天實際行為轉成方向：

[
A_{t,i}=
\begin{cases}
+1 & \text{Buy}\
+1 & \text{Hold，且實際持股}>0\
-1 & \text{Sell}\
0 & \text{無持股且無交易}
\end{cases}
]

這裡把 Buy 與 Hold 都視為：

> Agent 願意把股票留在 Portfolio 中。

Sell 則代表：

> Agent 想降低／移除這檔股票。

目前**不考慮股數**，所以：

[
Buy\ 1股 = Buy\ 100股
]

在這套簡化評分中都是：

[
A=+1
]

---

# 3. 單一股票當日分數

每一個「日期 × 股票」的分數：

[
\boxed{
S_{t,i}=R_{t,i}\times A_{t,i}
}
]

例如：

| Ranking | Action |               計算 | Score |
| ------- | ------ | ---------------: | ----: |
| Top1    | Buy    |       (5\times1) |    +5 |
| Top1    | Hold   |       (5\times1) |    +5 |
| Top1    | Sell   |    (5\times(-1)) |    -5 |
| Top3    | Buy    |       (3\times1) |    +3 |
| Bottom3 | Buy    |    ((-3)\times1) |    -3 |
| Bottom1 | Hold   |    ((-5)\times1) |    -5 |
| Bottom1 | Sell   | ((-5)\times(-1)) |    +5 |
| Others  | 任意行為   |      (0\times A) |     0 |

因此評分原則就是：

[
\boxed{
Buy/Hold\ Top \rightarrow 加分
}
]

[
\boxed{
Sell\ Bottom \rightarrow 加分
}
]

反過來：

[
\boxed{
Sell\ Top \rightarrow 扣分
}
]

[
\boxed{
Buy/Hold\ Bottom \rightarrow 扣分
}
]

---

# 4. 每日 Daily Score

第 (t) 天所有股票的分數相加：

[
\boxed{
D_t
===

\sum_{i=1}^{N}
R_{t,i}A_{t,i}
}
]

其中 (N) 是股票數量。

雖然有 30 檔股票，但 Others：

[
R=0
]

所以實際有影響的只有當天：

* Top5
* Bottom5

共 10 檔股票。

---

## 每天理論最高分

Top1～Top5 全部 Buy / Hold：

[
5+4+3+2+1=15
]

Bottom1～Bottom5 全部 Sell：

[
(-5)(-1)+(-4)(-1)+(-3)(-1)+(-2)(-1)+(-1)(-1)
=15
]

因此：

[
\boxed{
D_{\max}=15+15=30
}
]

---

## 每天理論最低分

如果完全做反：

Top5 全部 Sell：

[
-15
]

Bottom5 全部 Buy / Hold：

[
-15
]

因此：

[
\boxed{
D_{\min}=-30
}
]

所以每天：

[
\boxed{
-30\le D_t\le30
}
]

---

# 5. 多天／整年度分數

假設 Test 期間共有 (T) 個有效交易日。

先計算每天的：

[
D_1,D_2,\dots,D_T
]

再取平均：

[
\boxed{
\bar D
======

\frac{1}{T}
\sum_{t=1}^{T}D_t
}
]

因為每天都是：

[
-30\le D_t\le30
]

所以平均後依然：

[
\boxed{
-30\le\bar D\le30
}
]

使用平均而不是總和的好處是：

> 不同 Window 即使交易日數略有不同，也可以直接比較。

---

# 6. Overall Score：轉換成 0～100

為了讓教授更容易直接判斷結果，把：

[
[-30,+30]
]

線性轉換成：

[
[0,100]
]

公式：

[
\boxed{
OverallScore
============

\frac{\bar D+30}{60}\times100
}
]

因此：

[
\bar D=-30
\Rightarrow
OverallScore=0
]

[
\bar D=0
\Rightarrow
OverallScore=50
]

[
\bar D=+30
\Rightarrow
OverallScore=100
]

---

# 7. 完整流程

整個評分流程可以整理成：

```text
Future Ranking
      ↓
每天找出 Top1~Top5、Bottom1~Bottom5
      ↓
給 Ranking Score R
Top = +5 ~ +1
Bottom = -5 ~ -1
Others = 0
      ↓
取得 Agent 實際行為
      ↓
Buy  = +1
Hold = +1（實際有持股）
Sell = -1
None = 0
      ↓
每個股票計算
S(t,i) = R(t,i) × A(t,i)
      ↓
當天所有股票相加
Daily Score D(t)
      ↓
範圍：-30 ~ +30
      ↓
所有 Test 日 Daily Score 取平均
      ↓
Average Daily Score
      ↓
線性轉換成 0 ~ 100
      ↓
Overall Score
```

數學上就是三個主要方程式：

### Step 1：單股分數

[
\boxed{
S_{t,i}=R_{t,i}A_{t,i}
}
]

### Step 2：每日分數

[
\boxed{
D_t=\sum_{i=1}^{N}S_{t,i}
=========================

\sum_{i=1}^{N}R_{t,i}A_{t,i}
}
]

### Step 3：整段期間

[
\boxed{
\bar D
======

\frac{1}{T}
\sum_{t=1}^{T}D_t
}
]

### Step 4：Overall Score

[
\boxed{
OverallScore
============

\frac{\bar D+30}{60}\times100
}
]

---

## Score 最後可以這樣解讀

| Overall Score | 意義                        |
| ------------: | ------------------------- |
|        接近 100 | 大量做出符合 Future Ranking 的行為 |
|          > 50 | 整體交易方向偏正確                 |
|          = 50 | 正負行為互相抵銷／沒有明顯一致性          |
|          < 50 | 整體交易方向偏錯誤                 |
|          接近 0 | 幾乎完全與 Future Ranking 相反   |

所以之後例如比較：

| Reward       | Overall Score |
| ------------ | ------------: |
| FinRL_reward |          53.8 |
| DWFT n=40    |          61.4 |
| DWFT n=60    |      **68.7** |

就可以很簡單地說：

> **DWFT n=60 的 Agent 行為與未來股票排名具有較高的一致性。**

Summary:

* Future Ranking 定義為 Top1～Top5 = `+5～+1`，Bottom1～Bottom5 = `-5～-1`。
* Agent 行為定義為 Buy/Hold = `+1`、Sell = `-1`、無持股無交易 = `0`。
* 單股分數為
  [
  S_{t,i}=R_{t,i}A_{t,i}
  ]
* 每日分數為
  [
  D_t=\sum_iR_{t,i}A_{t,i}
  ]
  固定範圍 **-30～+30**。
* 多天先取 Daily Score 平均，再轉換成 **0～100 Overall Score**。
* 最終核心公式：
  [
  \boxed{OverallScore=\frac{\bar D+30}{60}\times100}
  ]

Modified files:

* 無。

How to test:

* 完美情況：Top5 全 Buy/Hold、Bottom5 全 Sell → Daily Score = `+30` → Overall = `100`。
* 完全相反 → `-30` → Overall = `0`。
* 正負完全抵銷 → `0` → Overall = `50`。

Warnings / Notes:

* 目前版本**完全不考慮資金量、交易股數與持股股數**。
* Hold 只有在「實際持股 > 0 且當天沒有 Buy/Sell」時才視為 `+1`。
* Others 的 Ranking Score 為 0，因此不影響 Overall Score。
