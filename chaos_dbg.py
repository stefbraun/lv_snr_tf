import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gate(W, U, x, h,b):
    gate_activation = sigmoid(np.dot(W, x) + np.dot(U, h) + b)
    return gate_activation

def activation(u, h, r, W, U, x, b):
    new_h = u * h + (1 - u) * np.tanh(np.dot(W, x) + np.dot(U, (r * h)) + b)
    helper=(1-u)*np.tanh(np.dot(W, x) + np.dot(U, (r * h)))
    helper2 = np.abs(helper)
    helper3=np.sum(helper2)
    return new_h

inp= np.zeros((2))
state= np.array([0, 1e-7])

W = dict()
W['r'] = np.array([[0, 0], [0, 0]])
W['u'] = np.array([[0, 0], [0, 0]])
W['c'] = np.array([[0, 0], [0, 0]])
U = dict()
U['r'] = np.array([[0, 0], [0, 0]])
U['u'] = np.array([[0, 0], [0, 0]])
U['c'] = np.array([[-5, -8], [8, 5]])
U['c'] = np.array([[5, 8], [-8, 5]])
# U['c'] = np.random.uniform(low=-2, high=5, size=(2,2))
U['c']=np.random.randint(5, size=(2,2))
U['c'][0,0]=-U['c'][0,0]
print(U['c'])
b=dict()
b['r'] = np.array([2.5,0.9])
b['u'] = np.array([-1,1])
b['c'] = np.array([1,2])

u_gate = gate(W=W['u'], U=U['u'], x=inp, h=state, b=b['u'])
r_gate = gate(W=W['r'], U=U['r'], x=inp, h=state, b=b['r'])
reg = []
state_reg = []
for k in np.arange(1,2,1):
    U['c'] = U['c']
    # state = np.array([0, 1e-7])

    for i in range(200):
        state = activation(u=u_gate, h=state, r=r_gate, W=W['c'], U=U['c'], x=inp, b=b['c'])
        state_reg.append(state)
        # print(state)

        # if np.sum(np.abs(state)) <=1e-10:
            # print(k)
            # break
    print(state)
    reg.append(np.sum(np.abs(state)))

plt.imshow(np.asarray(state_reg).T, aspect='auto', cmap = 'viridis', interpolation='none')
plt.colorbar()
plt.show()
5+5
