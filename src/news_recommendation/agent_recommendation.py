"""Agents for news recommendation problem."""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import random as rnd

from base.agent import Agent

_SMALL_NUMBER = 1e-10
_MEDIUM_NUMBER=.01
_LARGE_NUMBER = 1e+2
##############################################################################

class GreedyNewsRecommendation(Agent):
  """Greedy News Recommender."""
  
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001):
    """Args:
      num_articles - number of news articles
      dim - dimension of the problem
      theta_mean - mean of each component of theta
      theta_std - std of each component of theta
      epsilon - used in epsilon-greedy.
      alpha - used in backtracking line search
      beta - used in backtracking line search
      tol - stopping criterion of Newton's method.
      """
      
    self.num_articles = num_articles
    self.dim = dim
    self.theta_mean = theta_mean
    self.theta_std = theta_std
    self.back_track_alpha = alpha
    self.back_track_beta = beta
    self.tol = tol
    self.epsilon = epsilon
   
    # keeping current map estimates and Hessians for each news article
    # current_map_estimates: [num_articles, dim]
    self.current_map_estimates = [self.theta_mean*np.ones(self.dim) 
                                            for _ in range(self.num_articles)]
    # 每个article有一个Hessian
    # current_sessions:[num_articles, dim, dim]
    self.current_Hessians = [np.diag([1/self.theta_std**2]*self.dim) 
                                            for _ in range(self.num_articles)]
  
    # keeping the observations for each article
    self.num_plays = [0 for _ in range(self.num_articles)] #[arm, 1], 每个article的展示计数
    self.contexts = [[] for _ in range(self.num_articles)] #[arm, ]
    self.rewards = [[] for _ in range(self.num_articles)]
    
  def _compute_gradient_hessian_prior(self, x):
    '''computes the gradient and Hessian of the prior part of 
        negative log-likelihood at x.
        -[ylogp+(1-y)log(1-p)]
        '''
    Sinv = np.diag([1/self.theta_std**2]*self.dim) # 对角矩阵, [dim, dim]
    # mu:[dim,1]
    mu = self.theta_mean*np.ones(self.dim) # mu = theta*I

    # x:[dim, 1], mu:[dim, 1]
    g = Sinv.dot(x - mu) # H^(-1)* (x-mu), 不明白为何这样计算
    H = Sinv
    
    return g,H
  
  def _compute_gradient_hessian(self,x,article):
    """computes gradient and Hessian of negative log-likelihood  
    at point x, based on the observed data for the given article."""
    
    g,H = self._compute_gradient_hessian_prior(x)

    # 注意:此处的梯度与DL中的不太一样, 还有一个先验梯度
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i] # article feature, 即arm特征, [dim,1]
      y = self.rewards[article][i] #
      pred = 1/(1+np.exp(-x.dot(z))) # [dim,1]
      # 将所有梯度,Hession 进行累加
      g = g + (pred-y)*z # logistic regression 梯度更新
      H = H + pred*(1-pred)*np.outer(z,z) # hessian矩阵更新, Hessian += P(1-P) * np.outer(xi, xi)
    
    return g,H

  def _evaluate_log1pexp(self, x):
    """
    given the input x, returns log(1+exp(x)).
    log(1+exp(x)) = -log(1/(1+exp(x)))
    = -log(1+exp(x)-exp(x)/(1+exp(x)))
    = -log(1-sigmoid(x))
    即此函数与sigmoid正相关
    """
    if x > _LARGE_NUMBER:
      return x
    else:
      return np.log(1+np.exp(x))

  def _evaluate_negative_log_prior(self, x):
    """returning negative log-prior evaluated at x."""
    Sinv = np.diag([1/self.theta_std**2]*self.dim) 
    mu = self.theta_mean*np.ones(self.dim)
    
    return 0.5*(x-mu).T.dot(Sinv.dot(x-mu))

  def _evaluate_negative_log_posterior(self, x, article):
    """evaluate negative log-posterior for article at point x."""

    value = self._evaluate_negative_log_prior(x)
    # 先验*似然 = 后验, 此和为log域,所以是相加
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i] # z:[dim,1]
      y = self.rewards[article][i] #
      xz = x.dot(z) # float
      #
      value = value + self._evaluate_log1pexp(xz) - y*xz  # 此处不知是何意, 感觉像是对sigmoid预测概率的近似值
      
    return value
  
  def _back_track_search(self, x, g, dx, article):
    """Finding the right step size to be used in Newton's method.
    Inputs:
      x - current point
      g - gradient of the function at x
      dx - the descent direction

    Retruns:
      t - the step size"""

    step = 1
    current_function_value = self._evaluate_negative_log_posterior(x, article)
    difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
    (current_function_value + self.back_track_alpha*step*g.dot(dx))
    while difference > 0:
      step = self.back_track_beta * step # 缩小步长再搜索
      difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
          (current_function_value + self.back_track_alpha*step*g.dot(dx))

    return step

  def _optimize_Newton_method(self, article):
    """Optimize negative log_posterior function via Newton's method for the
    given article.

    严格的牛顿法
    """
    
    x = self.current_map_estimates[article] # x:[dim, 1]
    error = self.tol + 1
    while error > self.tol:
      g, H = self._compute_gradient_hessian(x,article)
      # 计算下降方向
      delta_x = -npla.solve(H, g) # H*p_k =g, p_k为x(k)的更新方向
      # 搜索步长
      step = self._back_track_search(x, g, delta_x, article)
      # 更新参数x
      x = x + step * delta_x
      error = -g.dot(delta_x) # 啥意思? 一般是计算||g||<epsilon吧,即梯度接近于0
      
    # computing the gradient and hessian at final point
    g, H = self._compute_gradient_hessian(x, article)

    # updating current map and Hessian for this article
    self.current_map_estimates[article] = x
    self.current_Hessians[article] = H
    return x, H
  
  def update_observation(self, context, article, feedback):
    '''updates the observations for displayed article, given the context and 
    user's feedback. The new observations are saved in the history of the 
    displayed article and the current map estimate and Hessian of this 
    article are updated right away.
    
    Args:
      context - a list containing observed context vector for each article
      article - article which was recently shown
      feedback - user's response.
      '''
    self.num_plays[article] += 1
    self.contexts[article].append(context[article])
    self.rewards[article].append(feedback)
    
    # updating the map estimate and Hessian for displayed article
    _,__ = self._optimize_Newton_method(article)
  
  def _map_rewards(self,context):
    map_rewards = [] # [arm, 1]
    for i in range(self.num_articles):
      # x:[dim,1]
      x = context[i]
      # theta:[dim, 1]
      theta = self.current_map_estimates[i] # 所有估计的theta
      map_rewards.append(1/(1+np.exp(-theta.dot(x)))) # reward = sigmoid(wx)
    return map_rewards
  
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    map_rewards = self._map_rewards(context) # greedy方法直接利用模型参数估计,然后选择一个最大的reward
    article = np.argmax(map_rewards)
    return article

##############################################################################
class EpsilonGreedyNewsRecommendation(GreedyNewsRecommendation):
  '''Epsilon greedy agent for the news recommendation problem.'''
  # epsilon算法则是一定概率随机选择
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    map_rewards = self._map_rewards(context)
    if np.random.uniform()<self.epsilon:
      article = np.random.randint(0,self.num_articles)
    else:
      article = np.argmax(map_rewards)
    return article

##############################################################################
class LaplaceTSNewsRecommendation(GreedyNewsRecommendation):   
  '''Laplace approximation to TS for news recommendation problem.'''
  def _sampled_rewards(self,context):
    sampled_rewards = []
    for i in range(self.num_articles): # 遍历每个action
      x = context[i]
      # current_map_estimates: [num_articles, dim]
      # mean:[dim,1]
      mean = self.current_map_estimates[i] # 计算每个action的theta
      cov = npla.inv(self.current_Hessians[i]) # 计算每个action的cov=Hessian^(-1)

      # 多元正态分布采样参数theta,注意:这里的参数不是确定的,而是采样出来的
      theta = np.random.multivariate_normal(mean, cov) # 采样theta
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x)))) # 重新估计每个action的回报
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    return article

##############################################################################
class LangevinTSNewsRecommendation(GreedyNewsRecommendation):
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001,batch_size = 100, step_count=200,
               step_size=.01):
    GreedyNewsRecommendation.__init__(self,num_articles,dim,theta_mean,theta_std,
              epsilon,alpha,beta,tol)
    self.batch_size = batch_size
    self.step_count = step_count
    self.step_size = step_size
    
  def _compute_stochastic_gradient(self, x, article):
    '''computes a stochastic gradient of the negative log-posterior for the given
     article.'''
    
    if self.num_plays[article]<=self.batch_size:
      sample_indices = range(self.num_plays[article])
      gradient_scale = 1
    else:
      gradient_scale = self.num_plays[article]/self.batch_size
      sample_indices = rnd.sample(range(self.num_plays[article]), self.batch_size)

    # 计算batch样本的梯度
    g = np.zeros(self.dim)
    for i in sample_indices:
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      pred = 1/(1+np.exp(-x.dot(z)))
      g = g + (pred-y)*z
    
    g_prior,_ = self._compute_gradient_hessian_prior(x)
    g = gradient_scale*g + g_prior
    return g
  
  def _Langevin_samples(self):
    '''gives the Langevin samples for each of the articles'''
    sampled_thetas = []
    for a in range(self.num_articles):
      # determining starting point and conditioner
      x = self.current_map_estimates[a]
      preconditioner = npla.inv(self.current_Hessians[a])
      preconditioner_sqrt=spla.sqrtm(preconditioner)
      
      #Remove any complex component in preconditioner_sqrt arising from numerical error
      complex_part=np.imag(preconditioner)
      if (spla.norm(complex_part)> _SMALL_NUMBER):
          print("Warning. There may be numerical issues.  Preconditioner has complex values")
          print("Norm of the imaginary component is, ")+str(spla.norm(complex_part))
      # 取实部
      preconditioner_sqrt=np.real(preconditioner_sqrt)
      # Monte Carol采样
      for i in range(self.step_count):
        g = -self._compute_stochastic_gradient(x,a)
        scaled_grad=preconditioner.dot(g)
        scaled_noise = preconditioner_sqrt.dot(np.random.randn(self.dim)) 
        x = x + self.step_size * scaled_grad+np.sqrt(2*self.step_size)*scaled_noise
      sampled_thetas.append(x)
    return sampled_thetas
  
  def _sampled_rewards(self,context):
    sampled_rewards = []
    sampled_theta = self._Langevin_samples()
    for i in range(self.num_articles):
      x = context[i]
      theta = sampled_theta[i]
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    return article
 