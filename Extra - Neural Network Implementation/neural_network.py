import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def propagate(x_1, x_2,target, iterations = 500, learning_rate = 1):

    # Initial Weights
    w_13 = 0.1
    w_14 = 0.4
    w_23 = 0.8
    w_24 = 0.6
    w_35 = 0.3
    w_45 = 0.9
    
    for i in range(1,iterations):
        input_1 = x_1
        input_2 = x_2

        # Hidden node 3
        print(f'\n--------Forward Pass {i}--------')
        a_1 = input_1 * w_13 + input_2 * w_23
        y3 = sigmoid(a_1)
        print('\n[ y3 =',y3,']')

        # Hidden node 4
        a_2 = input_1 * w_14 + input_2 * w_24
        y4 = sigmoid(a_2)
        print('[ y4 =',y4,']')

        # Output node 5
        a_3 = y3 * w_35 + y4 * w_45
        y5 = sigmoid(a_3)
        print('\n[ y5 =',y5,']')

        error = target - y5
        print('[ error =',error,']')
        
        if abs(error) < 0.01:
            print(f"\n-----------------------------\n\nTarget value {target} reached after {i} iterations.")
            return

        print(f'\n--------Backward Pass {i}--------')
        
        # Compute Deltas
        d5 = y5 * (1-y5) * error
        print('\n[ δ5 =',d5,']')

        d3 = y3 * (1-y3) * w_35 * d5
        print('[ δ3 =',d3,']')

        d4 = y4 * (1-y4) *  w_45 * d5
        print('[ δ4 =',d4,']')

        # Compute new weights - output node
        d_w_45 = learning_rate * d5 * y4

        print('\nW_45:',w_45,'-->', end=' ')
        w_45 += d_w_45
        print(w_45)

        d_w_35 = learning_rate * d5 * y3

        print('W_35:',w_35,'-->', end=' ')
        w_35 += d_w_35
        print(w_35)

        # Compute new weights - hidden node 3
        d_w_13 = learning_rate * d3 * input_1

        print('W_13:',w_13,'-->', end=' ')
        w_13 += d_w_13
        print(w_13)

        d_w_23 = learning_rate * d3 * input_2

        print('W_23:',w_23,'-->', end=' ')
        w_23 += d_w_23
        print(w_23)

        # Compute new weights - hidden node 4
        d_w_14 = learning_rate * d4 * input_1

        print('W_14:',w_14,'-->', end=' ')
        w_14 += d_w_14
        print(w_14)

        d_w_24 = learning_rate * d4 * input_2

        print('W_24:',w_24,'-->', end=' ')
        w_24 += d_w_24
        print(w_24)
        
#x_1 = 0.35
#x_2 = 0.9
#target = 0.5

x_1 = float(input('Enter x1: '))
x_2 = float(input('Enter x2: '))
target = float(input('Enter target value: '))

propagate(x_1,x_2,target)
