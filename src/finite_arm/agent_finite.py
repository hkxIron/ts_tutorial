"""Finite bandit agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from base.agent import Agent
from base.agent import random_argmax

_SMALL_NUMBER = 1e-10
##############################################################################


class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon # 以1-epilson选择贪婪
    # 每个arm的成功与失败数均初始化为1
    self.prior_success = np.array([a0 for arm in range(n_arm)]) # np.ones(n_arm)*a0,  [arm, 1]
    self.prior_failure = np.array([b0 for arm in range(n_arm)]) # [arm, 1]

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success) # [arm, 1]
    self.prior_failure = np.array(prior_failure) # [arm, 1]

  # 返回所有arm的后验分布
  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure) # [m,1]

  # 从后验分布中采样theta
  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure) # beta分布, 返回array:[arm, 1]

  # 更新后验分布
  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1): # element-wise接近1
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

  # 以1-epsilon的概率选择最优action
  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    if np.random.rand() < self.epsilon:
      action = np.random.randint(self.n_arm) # 从n个arm中随机选择一个
    else: # 1-epsilon greedy
      # 所谓reward, 就是success平均值
      posterior_means = self.get_posterior_mean() # shape:[arm, 1], 从中选择一个reward最大的arm
      action = random_argmax(posterior_means)

    return action

##############################################################################


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    # 注意: 只有此处不一样, 即TS里是从后验分布中采样,而epsilon-greedy是计算期望
    sampled_means = self.get_posterior_sample() # 每个arm都采样一个reward均值, [arm, 1]
    action = random_argmax(sampled_means) # 选择产生最大的均值的action
    return action


##############################################################################


class FiniteBernoulliBanditBootstrap(FiniteBernoulliBanditTS):
  """Bootstrapped Thompson sampling on finite armed bandit."""
  # 注意: BootstrappedTS里, 将后验采样改成了二项分布,而不是原来的Beta分布
  def get_posterior_sample(self):
    """Use bootstrap resampling instead of posterior sample."""
    total_tries = self.prior_success + self.prior_failure
    prob_success = self.prior_success / total_tries
    # np.random.binomial采样出来的是二项分布的均值, 即正面朝上的次数,所以要除以N
    boot_sample = np.random.binomial(total_tries, prob_success) / total_tries
    return boot_sample

##############################################################################


class FiniteBernoulliBanditLaplace(FiniteBernoulliBanditTS):
  """Laplace Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Gaussian approximation to posterior density (match moments)."""
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)

    """
    g(φ) denote a log-concave probability density function
    对于二项分布而言, 概率分布为:P(x)=c(n,r)*(x^r)*(1-x)^(n-r), x为变量
    lnP(x) = lnc(n,r) + r*lnx+(n-r)*ln(1-x)
    dlnP(x)/dx = r/x + (n-r)/(1-x)*(-1) = a/x - b/(1-x)
    d2lnP(x)/d2x = -a/x^2 - b/(1-x)^2
    
    此处x为众数,即:x = np = a/(a+b)
    
    众数（Mode）是指在统计分布上具有明显集中趋势点的数值，代表数据的一般水平。 
    也是一组数据中出现次数最多的数值，有时众数在一组数中有好几个
    在高斯分布中，众数位于峰值。
    """

    mode = a / (a + b) # 众数(对于连续分布), a:[arm,1], b: [arm, 1]
    hessian = a / mode + b / (1 - mode) # [arm, 1], TODO:此处是否计算有误?应该为 a/mode**2 + b/(1-mode)**2 ?
    """
    参见论文5.2:
    An approximate posterior sample θˆ is then drawn
    from a Gaussian distribution with mean θ and covariance matrix
    (−∇2 ln(ft−1(θ)))−1
    """
    laplace_sample = mode + np.sqrt(1 / hessian) * np.random.randn(self.n_arm) # 采样arm个样本
    return laplace_sample

##############################################################################


class DriftingFiniteBernoulliBanditTS(FiniteBernoulliBanditTS):
  """Thompson sampling on finite armed bandit."""

  def __init__(self, n_arm, a0=1, b0=1, gamma=0.01):
    self.n_arm = n_arm
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for arm in range(n_arm)]) # [arm, 1]
    self.prior_failure = np.array([b0 for arm in range(n_arm)])
    self.gamma = gamma

  """
  只在更新参数时与原来的模型不一样,一直使用初始的模型将老模型bound住, 进行decay的更新
  """
  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    # All values decay slightly, observation updated
    # 有点类似于深度学习中的参数与上一次的 加权平均
    self.prior_success = self.prior_success * (1 - self.gamma) + self.a0 * self.gamma
    self.prior_failure = self.prior_failure * (1 - self.gamma) + self.b0 * self.gamma
    #reward更新并不受影响
    self.prior_success[action] += reward
    self.prior_failure[action] += 1 - reward

##############################################################################
class FiniteBernoulliBanditLangevin(FiniteBernoulliBanditTS):
  '''Langevin method for approximate posterior sampling.'''
  
  def __init__(self,
               n_arm,
               step_count=100, # Monte Carlo采样次数
               step_size=0.01, # 样本融合速率
               a0=1,
               b0=1,
               epsilon=0.0):
    FiniteBernoulliBanditTS.__init__(self,n_arm, a0, b0, epsilon)
    self.step_count = step_count
    self.step_size = step_size
  
  def project(self,x):
    '''projects the vector x onto [_SMALL_NUMBER,1-_SMALL_NUMBER] to prevent
    numerical overflow.'''
    return np.minimum(1-_SMALL_NUMBER, np.maximum(x,_SMALL_NUMBER))

  """
  g(φ) denote a log-concave probability density function
  对于二项分布而言, 概率分布为:P(x)=c(n,r)*(x^r)*(1-x)^(n-r), x为变量
  lnP(x) = lnc(n,r) + r*lnx+(n-r)*ln(1-x)
  dlnP(x)/dx = r/x + (n-r)/(1-x)*(-1) = a/x - b/(1-x)
  
  d2lnP(x)/d2x = -a/x^2 - b/(1-x)^2
  
  """
  def compute_gradient(self, x):
    grad = (self.prior_success-1)/x - (self.prior_failure-1)/(1-x)
    return grad
    
  def compute_preconditioners(self, x):
    second_derivatives = (self.prior_success-1)/(x**2) + (self.prior_failure-1)/((1-x)**2) # (-1)*Hessian
    second_derivatives = np.maximum(second_derivatives,_SMALL_NUMBER)
    # A = - (H^-1)
    preconditioner = np.diag(1/second_derivatives) # 与论文中一致
    preconditioner_sqrt = np.diag(1/np.sqrt(second_derivatives)) # diag(1/sqrt(H))
    return preconditioner, preconditioner_sqrt
    
  def get_posterior_sample(self):
        
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    # 计算二项分布的概率均值
    x_map = a / (a + b)
    x_map = self.project(x_map)
    # paper:5.3节
    preconditioner, preconditioner_sqrt=self.compute_preconditioners(x_map)

    # 将x融合多次之后再采样, Monte Carlo 采样
    x = x_map
    for _ in range(self.step_count):
      g = self.compute_gradient(x)
      scaled_grad = preconditioner.dot(g) # 所谓的预条件数,只不过是 H^(-1), 即牛顿梯度法: x+= - H^(-1)*grad
      scaled_noise= preconditioner_sqrt.dot(np.random.randn(self.n_arm)) # 高斯噪声
      # phi = phi + epsilon*A* grad(ln(g(phi))) + sqrt(2*epsilon)*sqrt(A)*Wn
      x = x + self.step_size*scaled_grad + np.sqrt(2*self.step_size)*scaled_noise
      x = self.project(x)
      
    return x
      
    
    