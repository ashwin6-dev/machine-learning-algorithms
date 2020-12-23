class linear_regression:
  def __init__(self):
    self.m = 1
    self.c = 1

  def predict(self, x):
    return self.m * x + self.c

  def mean(self, arr):
    return sum(arr) / len(arr)

  def fit(self, x, y):
    #calculating gradient
    x_mul_y = [_x * _y for _x, _y in zip(x, y)]
    x_squared = [_x ** 2 for _x in x]
    self.m = (((self.mean(x) * self.mean(y)) - self.mean(x_mul_y))) / (self.mean(x) ** 2 - self.mean(x_squared))

    #calculating y-intercept
    self.c = y[0] - self.m * x[0]
