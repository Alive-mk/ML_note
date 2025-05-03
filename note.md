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

- cost function

<img src="pictures\image-20250503100057450.png" alt="image-20250503100057450" style="zoom: 80%;" />

`MSE` mean squared error cost function, 1/2是为了让后续计算更加简洁。
$$
minmize_{w,b}J(w,b)
$$
里面的