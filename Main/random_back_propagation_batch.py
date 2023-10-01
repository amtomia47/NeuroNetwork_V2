import numpy as np
import pandas as pd
import random

def compute_back_value_hidden(layer_dims,curren_layer,w,back_value):
    value = []
    node_partial = 0
    for l in range(layer_dims[curren_layer-1]):
        node_partial = 0
        for m in range(layer_dims[curren_layer]):
            node_partial += back_value[curren_layer] * w[curren_layer][l][m]
        value.append(node_partial)
    return value


input_len = np.loadtxt('Backup/input_len').astype(int)
layer_dims = np.loadtxt('Backup/layer_dims').astype(int)

name = open('Stock_data/stock_name','r').read()
stock = pd.read_csv('Stock_data/original/'+name+'10Y')
stock = list(stock.filter(['Close'])['Close'])
data_len = len(stock) ## 2456
data_total_set = np.arange(data_len) ## start form 0 to data_len-1
data_total_set = data_total_set[60:] ## start form 60 to 2515 , total 2456 , used for index the stock as input
random.shuffle(data_total_set)
training_set = []
## *****************隨機梯度下降 前置工作******************
index = 0
for i in range(344):
    training_set.append(data_total_set[index:index+5].tolist())
    index += 5

learn_rate = 0.1

validataion_set = []
while index < len(data_total_set):
    validataion_set.append(data_total_set[index])
    index+=1
# ***************create validation set*************************
np.savetxt('Backup/validation_set',np.array(validataion_set))

prediction = np.loadtxt('Prediction/'+name+'_PREDICT') ## useless?????????????
real = np.loadtxt('Stock_data/up_down/'+name+'_UP_DOWN').astype(int)
w = []
b = []
## load the value of parameters
w.append(np.loadtxt('WB/w/w1').tolist())
for i in range(len(layer_dims)-1):
    w.append(np.loadtxt('WB/w/w'+str(i+2)).tolist())
## tolist ?????????????????????????????????
count = 0.0
for i in training_set: ## iterate all batches
## iterate 344 times
    w_fix = []
    b_fix = []
    w_fix.append(np.zeros(input_len * layer_dims[0])) ## reshape ???
    for j in range(len(layer_dims)-1):
        w_fix.append( np.zeros(layer_dims[j] * layer_dims[j+1]))
    ## the correction amount of w of the batch
    for j in range(len(layer_dims)):
        b_fix.append(np.zeros(layer_dims[j]))
    ## the correction amount of y of the batch
    ## one dimension list

## *****修正量*****
    for j in i: ## iterate all elements in batch
        count +=1
        process_rate = count / 1720
        print("process back propagation: %.5f" % process_rate)
## iterate 5 times
        curren_layer = len(layer_dims)-1 ## index initial in reverse
        final_out = [0,1]
        if real[j-60] == 0:
            final_out[0] = 1
            final_out[1] = 0
        w_batch= []
        b_batch= []
        ## the correction amount of the batch
        for k in range(len(layer_dims)):
            if k == 0 :
                w_batch.append(np.zeros(layer_dims[0]*input_len))
            else:
                w_batch.append(np.zeros(layer_dims[k] * layer_dims[k-1]))
            b_batch.append(np.zeros(layer_dims[k]))
        back_value = []
        w_element = []
        b_element = []
        for k in range(len(layer_dims)):
            if k ==0 :
                w_element.append(np.zeros(input_len * layer_dims[0]))
            else:
                w_element.append(np.zeros(layer_dims[k] * layer_dims[k-1]))
            b_element.append(np.zeros(layer_dims[k]))
            ## the correction amount of single element in the batch
        for k in range(len(layer_dims)): ## iterate all layer of element
            w_tmp = []
            b_tmp = []
            ## correction amount of each layer in reverse order i.e. index 0 is layer 6
            layer_out = np.loadtxt('Compute_parameter/'+str(j+1-60)+'l'+str(curren_layer+1)+'_out')
            layer_original = np.loadtxt('Compute_parameter/'+str(j+1-60)+'l'+str(curren_layer+1)+'_z')
            if curren_layer>1:
                layer_in = np.loadtxt('Compute_parameter/'+str(j+1-60)+'l'+str(curren_layer-1)+'_out')

            if k == 0 : ## L6   sigmoid
                ## shape , 10 * 2
                for l in range(layer_dims[curren_layer]):
                    
                    for m in range(layer_dims[curren_layer-1]):
                        w_tmp.append(
                            -( 2*(final_out[l] - layer_out[l]) * layer_out[l] * (1-layer_out[l]) * layer_in[m] ) * learn_rate
                        )
                        ## partial W
                    b_tmp.append(
                        -(2*(final_out[l] - layer_out[l]) * layer_out[l] * (1-layer_out[l])) * learn_rate
                    )
                    ## partial B
                node_partial = 0
                for l in range(layer_dims[curren_layer-1]):
                    node_partial = 0
                    for m in range(layer_dims[curren_layer]):
                        node_partial +=  (-( 2*(final_out[m] - layer_out[m]) * layer_out[m] * (1-layer_out[m]) * w[curren_layer][l][m]))
                    back_value.append(node_partial)
                
            elif k==1: ## L5    sigmoid -> elu
                ## shape , 20 * 10
                
                for l in range(layer_dims[curren_layer]):
                    for m in range(layer_dims[curren_layer-1]):
                        if layer_original[l] < 0:
                            w_tmp.append(
                                back_value[l] * (layer_out[l]+1) * layer_in[m] * learn_rate
                            )
                        else :
                            w_tmp.append(
                                back_value[l] * layer_in[m] * learn_rate
                            )
                    if layer_original[l] < 0:
                        b_tmp.append(
                            back_value[l] * (layer_out[l]+1) * learn_rate
                        )
                    else:
                        b_tmp.append(
                            back_value[l] * learn_rate
                        )
                back_value = compute_back_value_hidden(layer_dims,curren_layer,w,back_value)
            elif k==2: ## L4    sigmoid -> elu -> tanh
                ## shape 30 * 20
                for l in range(layer_dims[curren_layer]):
                    for m in range(layer_dims[curren_layer-1]):
                        w_tmp.append(
                            back_value[l] * (1/np.cosh(layer_original[l]) ** 2) * layer_in[m] * learn_rate
                        )
                    b_tmp.append(
                        back_value[l] * (1/np.cosh(layer_original[l]) **2) * learn_rate
                    )
                back_value = compute_back_value_hidden(layer_dims,curren_layer,w,back_value)
            elif k==3: ## L3    sigmoid -> elu -> tanh -> sigmoid
                for l in range(layer_dims[curren_layer]):
                    for m in range(layer_dims[curren_layer-1]):
                        w_tmp.append(
                            back_value[l] * layer_out[l] * (1 - layer_out[l]) * layer_in[m] * learn_rate
                        )
                    b_tmp.append(
                        back_value[l] * layer_out[l] * (1 - layer_out[l]) * learn_rate
                    )
                back_value = compute_back_value_hidden(layer_dims,curren_layer,w,back_value)
            elif k==4: ## L2    sigmoid -> elu -> tanh -> sigmoid -> leacky_relu
                for l in range(layer_dims[curren_layer]):
                    for m in range(layer_dims[curren_layer-1]):
                        r = 0.0
                        if layer_original[l] > 0 :
                            r = 1
                        else:
                            r = 0.01
                        w_tmp.append(
                            back_value[l] * r * layer_in[m] * learn_rate
                        )
                    b_tmp.append(
                        back_value[l] * r * learn_rate
                    )
                back_value = compute_back_value_hidden(layer_dims,curren_layer,w,back_value)
            elif k==5: ## L1    sigmoid -> elu -> tanh -> sigmoid -> leacky_relu -> sigmoid
                nn_input = np.array(stock[j-60:j])
                nn_input = (nn_input - nn_input.mean()) / nn_input.std()
                for l in range(layer_dims[curren_layer]):
                    for m in range(input_len):
                        w_tmp.append(
                            back_value[l] * layer_out[l] * (1 - layer_out[l]) * nn_input[m] * learn_rate
                        )
                    b_tmp.append(
                        back_value[l] * layer_out[l] * (1 - layer_out[l]) * learn_rate
                    )
            w_element[curren_layer] += np.array(w_tmp)
            b_element[curren_layer] += np.array(b_tmp)
            curren_layer-=1
        ## all elements in batch i.e. len(w) = 5
        for k in range(len(layer_dims)):
            w_batch[k] += w_element[k]/len(i)
            b_batch[k] += b_element[k]/len(i)
    for j in range(len(layer_dims)):
        w_fix[j] += w_batch[j]
        b_fix[j] += b_batch[j]
w_new = np.array(w[0]).reshape(input_len , layer_dims[0])
w_new += w_fix[0].reshape(input_len , layer_dims[0])
np.savetxt('WB/w/w1',w_new)
for i in range(len(layer_dims)-1):
    w_new = np.array(w[i+1]).reshape(layer_dims[i],layer_dims[i+1])
    w_new += w_fix[i+1].reshape(layer_dims[i],layer_dims[i+1])
    np.savetxt('WB/w/w'+str(i+2),w_new)
for i in range(len(layer_dims)):
    b = np.loadtxt('WB/b/b'+str(i+1)).tolist()
    b_new = np.array(b)
    b_new += b_fix[i]
    np.savetxt('WB/b/b'+str(i+1),b_new)