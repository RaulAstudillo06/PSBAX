#%%
import torch
import math
import numpy as np
import botorch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
torch.set_default_dtype(torch.float64)
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


#%%

# one dimensional gaussian

mean1d = 0
std1d = 1
def f1d(x, mean=mean1d, std=std1d):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(torch.tensor(2 * torch.pi)))


x_axis = torch.linspace(-3, 3, 1000)
y_axis = f1d(x_axis)
fig, ax = plt.subplots(1, 1)
ax.plot(x_axis, y_axis)
plt.xticks([])
plt.yticks([])
plt.savefig('1d_density.png', dpi=300)
plt.show()
#%%

# mvn = torch.distributions.MultivariateNormal(mean1d, torch.diag(std1d ** 2))
# x1d = torch.linspace(-3, 3, 1000)
# n1d = 10
# y1d = np.random.standard_normal(n1d)

# x1d = np.full_like(
#     y1d,
#     0.5
# )

# # plot y1d
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# scatter = ax.scatter(x1d, y1d, c=y1d, cmap='viridis')
# plt.xticks([])
# plt.yticks([])
# plt.show()
    
# two dimensional gaussian
# %%
mean, cov = [0., 0.], [(1., 0.), (0., 1.)]
data = np.random.multivariate_normal(mean, cov, 1000)
#%%
n = 1 # dimension
m = 10 # samples

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, cov, m).T

plt.clf()

fig, ax = plt.subplots(1, 1)
#plt.plot(Xshow, f_prior, '-o')
Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    ax.plot(Xshow, f_prior, '-o', linewidth=1)
plt.xticks([])
plt.yticks([])
# plt.title('10 samples of the 20-D gaussian prior')
plt.savefig('1d.png', dpi=300)
plt.show()
# %%
n = 2 # dimension
m = 10 # samples

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, cov, m).T

plt.clf()

#plt.plot(Xshow, f_prior, '-o')
Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, '-o', linewidth=1)
    
# plt.title('10 samples of the 20-D gaussian prior')
plt.xticks([])
plt.yticks([])
plt.savefig('2d.png', dpi=300)
plt.show()
# %%
n = 20 
m = 10

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, cov, m).T

plt.clf()

#plt.plot(Xshow, f_prior, '-o')
Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, '-o', linewidth=1)
    
# plt.title('10 samples of the 20-D gaussian prior')
plt.xticks([])
plt.yticks([])
plt.savefig('2d.png', dpi=300)
plt.show()

#%%
def kernel(a, b):
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # np.sum( ,axis=1) means adding all elements columnly; .reshap(-1, 1) add one dimension to make (n,) become (n,1)
    return np.exp(-.5 * sqdist)

n = 200  
m = 10

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

K_ = kernel(Xshow, Xshow)                  # k(x_star, x_star)        

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, K_, m).T

plt.clf()

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, '-o', linewidth=1)
plt.xticks([])
plt.yticks([])
plt.savefig('20d_kernel.png', dpi=300)
plt.show()
# %%

n = 200  
m = 10

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

def kernel(a, b, lengthscale=0.05):
    sqdist = np.sum(a**2,axis=1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
   
    return np.exp(-.5 * sqdist / lengthscale)

K_ = kernel(Xshow, Xshow)                  # k(x_star, x_star)        

mean = np.zeros(n)
cov = np.eye(n)

f_prior = np.random.multivariate_normal(mean, K_, m).T

plt.clf()
plt.figure(figsize=(10, 5))

Xshow = np.linspace(0, 1, n).reshape(-1,1)   # n number test points in the range of (0, 1)

for i in range(m):
    plt.plot(Xshow, f_prior, linewidth=1)
plt.xticks([])
plt.yticks([])
plt.savefig('200d_kernel.png', dpi=300)
plt.show()
# %%
