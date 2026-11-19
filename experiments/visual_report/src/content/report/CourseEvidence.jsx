import React, { useState } from 'react';

const evidenceURL = 'https://github.com/ZhiChao-Chang/chemgpt-rl-study/blob/main/experiments/';

export function SearchExample() {
  const [attempts, setAttempts] = useState(100);
  const hitChance = 1 - 0.99 ** attempts;
  return <section className="r-callout" id="search-example">
    <h3>抽一次，还是保留多次抽样中的最高分？</h3>
    <p>假设策略 A 每次得 0.6 分。策略 B 有 1% 的机会得满分，其余时候得零。只抽一次，A 的期望分数是 0.6，B 只有 0.01。如果可以反复抽样，每次只保留目前最高分，你会选哪个？</p>
    <label>独立抽样次数 <select value={attempts} onChange={event => setAttempts(Number(event.target.value))}>
      {[1, 10, 50, 100, 500].map(n => <option key={n} value={n}>{n}</option>)}
    </select></label>
    <p aria-live="polite">抽 {attempts} 次，A 的最高分始终是 <strong>0.600</strong>。B 至少命中一次的概率为 {(hitChance * 100).toFixed(1)}%，期望最高分为 <strong>{hitChance.toFixed(3)}</strong>。此时按期望最高分选择策略 {hitChance > 0.6 ? 'B' : 'A'}。</p>
    <p>这个固定策略的小例子解释了我们为什么做两种评价。真实实验一边测训练结束后新生成分子的平均分，一边记录每发现一个新分子时，历史前十名的平均分。<a href={evidenceURL + 'evidence/objective_mismatch.json'}>查看例子计算</a></p>
  </section>;
}

export function CourseUpdate({ theory }) {
  return <section id="policy-update" className="r-teaching">
    <h3>从一条分子的终局得分，到一次策略更新</h3>
    <p><strong>第 3、4 讲 · 状态与动作。</strong>从起始符出发，ChemGPT 根据已经生成的完整前缀选择下一个 token，再把它追加到序列。两个前缀即使以同一个 token 结尾，也可能对应不同的化学结构。保留完整前缀，才能决定接下来有哪些有价值的选择。</p>
    <p><strong>第 6 讲 · 蒙特卡洛回报。</strong>一条序列结束后，我们解码分子并计算性质分数。这次完整采样给出一个回报。基础任务采用终局奖励和折扣因子 1，因此轨迹上每一步都使用同一个终局分数。若把折扣调低，即使最终分数相同，较早结束的轨迹也会获得更高的起点回报。</p>
    <p><strong>第 11 讲 · REINFORCE 与基线。</strong>通常一次生成 32 条轨迹。对于其中一条，用它的回报减去其余 31 条的平均回报，再用这个差值加权该轨迹各步的对数概率梯度。正差值推动策略提高这些动作的概率，负差值给出相反方向。每个新批次只更新一次。</p>
    <p>KL 条件沿用同一更新。计算轨迹回报时，先从 MPO 中扣除偏离初始策略的成本，再与批内基线比较。这样可以单独观察，引入先验约束后，生成长度、平均奖励和搜索效率分别发生了什么变化。</p>
    <h4>为什么要从基线里排除自己？</h4>
    <p>我们把问题缩到两步，穷举全部 {theory.trajectory_count} 条轨迹，先算出精确梯度，再比较不同估计量。当批量为 2 时，如果把自己也算进平均基线，自己的回报差就会缩小一半。在全部 {theory.independent_batches} 种有序批次上取期望，梯度恰好只剩精确值的 {theory.self_including_batch_mean_gradient_ratio * 100}%。排除自己后，最大绝对误差为 {theory.loo_error.toExponential(1)}。这项小实验直接检验了训练代码中基线的选择。 <a href={evidenceURL + 'src/check_theory.py'}>查看枚举代码</a></p>
  </section>;
}

export function SeedComparison({ runs, left, right, label }) {
  const reference = new Map(runs.filter(run => run.condition === right).map(run => [run.seed, run]));
  const pairs = runs.filter(run => run.condition === left && reference.has(run.seed))
    .map(run => ({ seed: run.seed, delta: run.auc - reference.get(run.seed).auc }));
  if (!pairs.length) return null;
  const positive = pairs.filter(pair => pair.delta > 0).length;
  const negative = pairs.filter(pair => pair.delta < 0).length;
  return <aside className="r-callout">
    <h4>同一随机种子下，两种方法的搜索 AUC 相差多少？</h4>
    <p>{label}。{pairs.length} 组中，差值为正的有 {positive} 组，为负的有 {negative} 组。每组对应同一任务和预算下的两次运行。</p>
    <ul>{pairs.map(({ seed, delta }) => <li key={seed}>种子 {seed}　{delta > 0 ? '+' : ''}{delta.toFixed(5)}</li>)}</ul>
  </aside>;
}
