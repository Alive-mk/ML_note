## concept

- 机器学习的定义

**机器学习**就是让计算机从**数据**中**自动提炼规律**，并具备在**新场景**中**应用**这些规律的能力。

- supervised learning
  - Regression
  - classification
- unsupervised learning
  - clustering
  - anomaly detection
  - dimensionality reduction

## linear regression model

### `MSE`

- cost function

<img src="pictures\image-20250503100057450.png" alt="image-20250503100057450" style="zoom: 80%;" />

`MSE` mean squared error cost function, 1/2是为了让后续计算更加简洁。
$$
minmize_{w,b}J(w,b)
$$

```python
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Use the browser renderer instead of notebook mime
pio.renderers.default = 'browser'

# Generate sample data
np.random.seed(0)
x = np.linspace(0, 10, 20)
y = 2 * x   + 1 + np.random.randn(*x.shape) * 2

# Define parameter ranges
w0_range = np.linspace(-1, 5, 100)
w1_range = np.linspace(-1, 4, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)

# Compute MSE loss
n = x.shape[0]
loss = (1/(2*n)) * np.sum((y.reshape(1,1,-1) - (W1[:,:,None] * x + W0[:,:,None]))**2, axis=2)

# Create interactive 3D surface
fig = go.Figure(data=[go.Surface(x=W0, y=W1, z=loss)])
fig.update_layout(
    title='Interactive MSE Loss Surface',
    scene=dict(
        xaxis_title='Intercept w0',
        yaxis_title='Slope w1',
        zaxis_title='MSE Loss'
    ),
    autosize=True,
    width=700,
    height=700
)

# Instead of fig.show(), write to an HTML file and open in browser
html_path = 'mse_surface.html'
fig.write_html(html_path, auto_open=True)
print(f"Interactive plot saved to {html_path} and opened in your default browser.")

```

<img src="pictures\image-20250503104023193.png" alt="image-20250503104023193" style="zoom:67%;" />

<img src="pictures\image-20250503104138502.png" alt="image-20250503104138502" style="zoom: 67%;" />

- convex function

### gradient descent

- 局部最优解
- 同步更新参数
- 学习率的选择

- 可视化

<img src="pictures\image-20250503143250252.png" alt="image-20250503143250252" style="zoom:67%;" />

- batch gradient descent：在每一次参数更新中，**使用整个训练集**来计算损失函数对参数的梯度，然后统一更新参数。

🟦 **Batch Gradient Descent（蓝色）**

- 路径平滑、方向精准
- 每次用所有数据更新参数，最稳定但慢

🔴 **Stochastic Gradient Descent – 带强噪声（红色）**

- 路径剧烈波动，甚至偏离目标方向
- 每次只用一个样本，有很强的“跳跃感”
- 速度快，但收敛过程不稳定

🟩 **Mini-batch Gradient Descent（绿色）**

- 路径略有抖动，但趋于稳定
- 折中了效率与稳定性，实际中最常用（如批量大小为 32、64、128）

<img src="pictures\image-20250503144557889.png" alt="image-20250503144557889" style="zoom: 50%;" />



:red_circle: header=None 与 header=0

- **`header=0` =“我有标题，行号 0 就是列名”。**

- **`header=None` =“我没有标题，所有行都是数据”。**

- **`skiprows=1`** 丢弃第一行的数据

:large_blue_diamond: 切片以及索引区别

- `cols-1:cols`（切片） → 返回 `DataFrame`，形状 `(n_rows, 1)`；

- `cols-1`（标量下标） → 返回 Series，形状 `(n_rows,)`。

<img src="pictures\image-20250503170011238.png" alt="image-20250503170011238" style="zoom:67%;" />