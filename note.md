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

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
NumPy 中用于创建在指定范围内等间距数值序列的函数
start：序列的起始值。
stop：序列的结束值。
num：（可选）生成的样本数量，默认为 50。
endpoint：（可选）如果为 True（默认），则包含终点 stop；如果为 False，则不包含。
retstep：（可选）如果为 True，则返回一个元组 (样本数组, 步长)。
dtype：（可选）指定输出数组的数据类型。
axis：（可选）指定生成数组的轴，默认是 0 轴。

np.random.randn(d0, d1, ..., dn)
NumPy 中用于生成标准正态分布（均值为0，标准差为1）的随机数的函数。
d0, d1, ..., dn：指定输出数组的维度。如果是 np.random.randn(2, 3)，则生成一个 2 行 3 列的二维数组。

np.random.seed(0)
设定随机数的种子

x.shape
x.reshape(2,-1)


```

