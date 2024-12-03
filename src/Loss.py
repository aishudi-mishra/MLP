import numpy as np
class Cross_entropy_loss:
  def __call__(self, y_pred=None, y=None):
    # Extrem cases might result in infinity
    if y_pred == 0.0:
      y_pred = 0.0001
    if y_pred == 1.0:
      y_pred = 0.9999

    loss = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
    return loss

  def gradient(self, y_pred, y):
    # Extrem cases might result in infinity
    if y_pred == 0.0:
      y_pred = 0.0001
    if y_pred == 1.0:
      y_pred = 0.9999

    return -y/y_pred + (1-y)/(1-y_pred)