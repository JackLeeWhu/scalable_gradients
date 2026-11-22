def simulate_gbm(u, sigma, seed_number, n, z0=1.0):
    """从 z0 出发，在 [0, 1] 上逐步模拟 GBM，画图并返回各条路径。"""
    import numpy as np
    import matplotlib.pyplot as plt


    # 一共 n 个区间、n+1 个时点。每个区间只抽一次布朗增量。
    dt = 1.0 / n
    time = np.linspace(0.0, 1.0, n + 1)
    rng = np.random.default_rng(seed_number)
    dw = rng.normal(0.0, np.sqrt(dt), size=n)

    # 正向 Itô Euler：在每段的左端点计算漂移和扩散，再加到当前值上。
    forward = np.empty(n + 1)
    forward[0] = z0
    for i in range(n):
        z = forward[i]
        forward[i + 1] = z + u * z * dt + sigma * z * dw[i]

    # 两条反向路径都从正向终点出发，第一步复用最后一个 dw。
    ito_reverse = np.empty(n + 1)
    strat_reverse = np.empty(n + 1)
    ito_reverse[-1] = strat_reverse[-1] = forward[-1]

    strat_drift = u - 0.5 * sigma**2
    for i in reversed(range(n)):
        # Itô 直接变号：回程从右端点出发，在右端点计算两个系数。
        z_right = ito_reverse[i + 1]
        ito_reverse[i] = z_right - u * z_right * dt - sigma * z_right * dw[i]

        # Stratonovich 中点步：z_left = z_right - step * (z_left + z_right)/2。
        # 两边都有 z_left。把它移到左边，就能算出这一期的值。
        z_right = strat_reverse[i + 1]
        step = strat_drift * dt + sigma * dw[i]
        if 1 + step / 2 == 0:
            raise ValueError(f"第 {i + 1} 个区间的中点方程无唯一解，请增大 n")
        strat_reverse[i] = z_right * (1 - step / 2) / (1 + step / 2)

    # 黑线稍宽，蓝色点线与它重合时，两条线都能看见。
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(time, forward, color="black", lw=2.6, label="Forward GBM")
    ax.plot(time, ito_reverse, "r--", label="Naive Itô reverse")
    ax.plot(time, strat_reverse, color="deepskyblue", ls=":", lw=2,
            label="Stratonovich reverse")
    ax.set(xlabel="Time", ylabel="Z(t)", title="GBM on a shared Brownian path")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()

    return time, dw, forward, ito_reverse, strat_reverse


if __name__ == "__main__":
    simulate_gbm(u=0.10, sigma=0.30, seed_number=42, n=250, z0=100)
