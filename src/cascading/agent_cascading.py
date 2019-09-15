"""Agents for cascading bandit problems.
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from base.agent import Agent

##############################################################################


class CascadingBanditEpsilonGreedy(Agent):

  def __init__(self, num_items, num_positions, a0=1, b0=1, epsilon=0.0):
    """An agent for cascading bandits.

    Args:
      num_items - "L" in math notation
      num_positions - "K" in math notation
      a0 - prior success
      b0
    """
    self.num_items = num_items
    self.num_positions = num_positions
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for item in range(num_items)]) # [arm, 1]
    self.prior_failure = np.array([b0 for item in range(num_items)]) # [arm, 1]
    self.epsilon = epsilon
    self.timestep = 1

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success) # [arm, 1]
    self.prior_failure = np.array(prior_failure) # [arm, 1]

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  # 采样
  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    """Updates observations for cascading bandits.

    Args:
      observation - tuple of (round_failure, round_success) each lists of items
      action - action_list of all the actions tried prior round
      reward - success or not
    """
    for action in observation['round_failure']:
      self.prior_failure[action] += 1

    for action in observation['round_success']:
      self.prior_success[action] += 1

    # Update timestep for UCB agents
    self.timestep += 1

  def pick_action(self, observation):
    # 从0~num_items个物品中采样 num_positions个用于展示
    if np.random.rand() < self.epsilon:
      action_list = np.random.randint(low=0, high=self.num_items, size=self.num_positions)
    else:
      # 选取最优值
      posterior_means = self.get_posterior_mean() # [arm, 1]
      action_list = posterior_means.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


def _ucb_1(empirical_mean, timestep, count):
  """Computes UCB1 upper confidence bound.

  Args:
    empirical_mean - empirical mean
    timestep - time elapsed
    count - number of visits to that object
  """
  confidence = np.sqrt((1.5 * np.log(timestep)) / count)
  return empirical_mean + confidence


class CascadingBanditUCB1(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    # 获取后验概率
    posterior_means = self.get_posterior_mean()
    ucb_values = np.zeros(self.num_items)
    for item in range(self.num_items):
      count = self.prior_success[item] + self.prior_failure[item] # 当前action被访问的次数
      ucb_values[item] = _ucb_1(posterior_means[item], self.timestep, count)

    # 选择ucb最大的k个arm
    action_list = ucb_values.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


def _kl_ucb(empirical_mean, timestep, count, tolerance=1e-3, maxiter=25):
  """Computes KL-UCB via binary search(二分查找)
  从 emprical_mean右边开查找第一个大于kl_bound的点

  Args:
    empirical_mean - empirical mean
    timestep - time elapsed
    count - number of visits to that object
    tolerance - accuracy for numerical bisection
    maxiter - maximum number of iterations
  """
  kl_bound = (np.log(timestep) + 3 * np.log(np.log(timestep + 1e-6))) / count
  upper_bound = 1
  lower_bound = empirical_mean

  # Most of the experiment is spent for small values of KL --> biased search
  n_iter = 0
  while (upper_bound - lower_bound) > tolerance:
    n_iter += 1
    midpoint = (upper_bound + 3 * lower_bound) / 4
    dist = _d_kl(empirical_mean, midpoint)

    if dist < kl_bound:
      lower_bound = midpoint # 向右查找
    else:
      upper_bound = midpoint # 向左查找

    if n_iter > maxiter:
      print(
          'WARNING: maximum number of iterations exceeded, accuracy only %0.2f'
          % (upper_bound - lower_bound,))
      break

  return lower_bound

# 计算kl距离
def _d_kl(p, q, epsilon=1e-6):
  """Compute the KL divergence for single numbers.
  KL(p||q) = plog(p/q)+(1-p)log((1-p)/(1-q))
  """
  if p <= epsilon: # p太小,则为0, limit 0log0 = 0
    A = 0
  else: # p不小,但q太小,则为inf
    A = np.inf if q <= epsilon else p * np.log(p / q)

  if p >= 1 - epsilon: # p太大,则为0
    B = 0
  else:
    B = np.inf if q >= 1 - epsilon else (1 - p) * np.log((1 - p) / (1 - q))

  return A + B


class CascadingBanditKLUCB(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    posterior_means = self.get_posterior_mean() # [arm, 1]
    ucb_values = np.zeros(self.num_items) # [arm, 1]
    for item in range(self.num_items):
      # 展示次数
      count = self.prior_success[item] + self.prior_failure[item]
      ucb_values[item] = _kl_ucb(posterior_means[item], self.timestep, count)

    action_list = ucb_values.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


class CascadingBanditTS(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    posterior_sample = self.get_posterior_sample() # [arm, 1]
    action_list = posterior_sample.argsort()[::-1][:self.num_positions]
    return action_list
