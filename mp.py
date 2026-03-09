import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-20,20,200)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
probility = sigmoid(x)
plt.figure(figsize=(10,6))
plt.plot(x,probility,label= 'sigma function',color='blue',linewidth=2)
plt.axhline(y=0.5,color='r',linestyle='--',label='Decision Threshold (0.5)')
plt.axvline(x=0, color='g', linestyle='--', label='z = 0')
plt.xlabel('Linear Output (z = wx + b)')
plt.ylabel('Probability P(Class=1)')
plt.title('How Sigmoid Squashes Linear Output to [0, 1]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.show()

print(f"z = -5.0  →  P = {sigmoid(-5):.4f}  →  Predict Class 0")
print(f"z =  0.0  →  P = {sigmoid(0):.4f}  →  Uncertain")
print(f"z =  5.0  →  P = {sigmoid(5):.4f}  →  Predict Class 1")