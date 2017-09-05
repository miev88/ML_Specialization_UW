import numpy as np

'''
x = [-2, -1, 1, 4]
y_true = [-3, -1, 2, 3]

x = np.asarray(x)
y_true = np.asarray(y_true)

y_pred = (41/42 * x) - 5/21

mean_y_true = y_true.mean()

exp_var = sum((y_pred - mean_y_true)**2)
unexp_var = sum((y_true - y_pred)**2)
tot_var = exp_var + unexp_var

print(exp_var/tot_var) #0.879644165358
'''



x = [1, 4]
y_true = [1, 4]

x = np.asarray(x)
y_true = np.asarray(y_true)

y_pred = x

mean_y_true = y_true.mean()

# explained and total variation
exp_var = sum((y_pred - mean_y_true)**2)
unexp_var = sum((y_true - y_pred)**2)
tot_var = exp_var + unexp_var

print('RSS :', unexp_var)
print('Standard error of the estimate: ', np.sqrt(unexp_var/len(y_true)))
print('R-squared :', exp_var/tot_var) #0.000823882911699
