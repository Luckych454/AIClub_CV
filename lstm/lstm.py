import numpy as np
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    #x-max_x避免超大数据的出现
    return np.exp(x-max_x)/np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x));

class toyLSTM:
    def __init__(self,input_dim,hidden_dim=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #初始化权重向量
        self.whi, self.wxi, self.bi = self.init_wh_wx()
        self.whf, self.wxf, self.bf = self.init_wh_wx()
        self.who, self.wxo, self.bo = self.init_wh_wx()
        self.wha, self.wxa, self.ba = self.init_wh_wx()
        self.wy = np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(self.input_dim,self.hidden_dim))
        self.by = np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(self.input_dim,1))

    #初始化 wh,wx,b
    def init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.hidden_dim));
        wx = np.random.uniform(-np.sqrt(1.0/self.input_dim),np.sqrt(1.0/self.input_dim),(self.hidden_dim,self.input_dim));
        b = np.random.uniform(-np.sqrt(1.0/self.input_dim),np.sqrt(1.0/self.input_dim),(self.hidden_dim,1))
        return wh,wx,b

    def init_state(self,T):
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T))
        ys = np.array([np.zeros((self.input_dim, 1))] * (T))
        return {
            'iss': iss, 'fss': fss, 'oss': oss,
            'ass': ass, 'hss': hss, 'css': css,
            'ys': ys
        }
        '''iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        ys = np.array([np.zeros((self.input_dim,1))]*(T+1))
        return {
            'iss':iss,'fss':fss,'oss':oss,
            'ass':ass,'hss':hss,'css':css,
            'ys':ys
        }'''

    def forward(self,x):
        tt = len(x)
        states = self.init_state(tt)
        for t in range(tt):
            ht_pre = np.array(states['hss'][t-1]).reshape(-1,1)

            #input gate
            states['iss'][t] = self.cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
            states['fss'][t] = self.cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
            states['oss'][t] = self.cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
            states['ass'][t] = self.cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)

            #cell state
            states['css'][t] = states['fss'][t]*states['css'][t-1] + states['iss'][t]*states['ass'][t]

            #hidden state
            states['hss'][t] = states['oss'][t]*tanh(states['css'][t])

            #output state
            states['ys'][t] = softmax(self.wy.dot(states['hss'][t])+self.by)
        return states


    def cal_gate(self,wh,wx,b,ht_pre,x,actication):
        #print(type(x))
        return actication(wh.dot(ht_pre)+wx[:, np.int(x)].reshape(-1, 1)+b)


    def predict(self,x):
        states = self.forward(x)
        #pre_y = np.argmax(states['ys'].reshape(len(x),-1),axis = 1)
        pre_y = states['hss'].reshape(len(x), -1)
        for i in range(len(y)):
            for j in range(len(y[i])):
                if pre_y[i][j]>0:
                    pre_y[i][j] = 1
                else:
                    pre_y[i][j] = 0
        return pre_y

    def loss(self,x,y):
        cost = 0
        for i in range(len(y)):
            states = self.forward(x[i])
            #pre_yi = states['ys'][range(len(y[i])),y[i]]
            pre_yi = states['ys'][range(len(y[i]))]
            #print(pre_yi.shape)
            #cost -= np.sum(np.log(pre_yi))
            cost -= np.sum(np.log(pre_yi))
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost/N
        return ave_loss

    def init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)
        return dwh,dwx,db

    def bptt(self,x,y):
        dwhi, dwxi, dbi = self.init_wh_wx_grad()
        dwhf, dwxf, dbf = self.init_wh_wx_grad()
        dwho, dwxo, dbo = self.init_wh_wx_grad()
        dwha, dwxa, dba = self.init_wh_wx_grad()
        dwy,dby = np.zeros(self.wy.shape),np.zeros(self.by.shape)
        delta_ct = np.zeros((self.hidden_dim,1))
        #前向计算
        states = self.forward(x)
        delta_o = states['ys']
        #delta_o[np.arange(len(y)), np.int(y)] -= 1
        delta_o[np.arange(len(y))] -= 1
        for t in np.arange(len(y))[::-1]:
            dwy += delta_o[t].dot(states['hss'][t].reshape(1,-1))
            dby += delta_o[t]
            delta_ht = self.wy.T.dot(delta_o[t])

            #各个门的偏导数
            delta_ot = delta_ht * tanh(states['css'][t])
            delta_ct += delta_ht * states['ass'][t] * (1-tanh(states['css'][t])**2)
            delta_it = delta_ct * states['ass'][t]
            delta_ft = delta_ct * states['css'][t]
            delta_at = delta_ct * states['iss'][t]

            delta_at_net = delta_at * (1-states['ass'][t]**2)
            delta_it_net = delta_it * states['iss'][t] * (1-states['iss'][t])
            delta_ft_net = delta_ft * states['fss'][t] * (1-states['fss'][t])
            delta_ot_net = delta_ot * states['oss'][t] * (1-states['oss'][t])

            #更新各矩阵的偏导数
            dwhf, dwxf, dbf = self.cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, states['hss'][t - 1], x[t])
            dwha, dwxa, dba = self.cal_grad_delta(dwha, dwxa, dba, delta_at_net, states['ass'][t - 1], x[t])
            dwhi, dwxi, dbi = self.cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, states['iss'][t - 1], x[t])
            dwho, dwxo, dbo = self.cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, states['oss'][t - 1], x[t])
        return [dwhf,dwxf,dbf,
                dwha,dwxa,dba,
                dwhi,dwxi,dbi,
                dwho,dwxo,dbo,
                dwy,dby]


    def cal_grad_delta(self,dwh,dwx,db,delta_net,ht_pre,x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net
        return dwh,dwx,db

    def sgd_step(self,x,y,learning_rate):
        dwhf, dwxf, dbf,\
        dwha, dwxa, dba,\
        dwhi, dwxi, dbi,\
        dwho, dwxo, dbo,\
        dwy, dby = self.bptt(x,y)

        self.whf, self.wxf, self.bf = self.update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.wha, self.wxa, self.ba = self.update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.whi, self.wxi, self.bi = self.update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.who, self.wxo, self.bo = self.update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)
        self.wy,self.by = self.wy-learning_rate*dwy,self.by - learning_rate*dby

    def update_wh_wx(self,learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate*dwh
        wx -= learning_rate*dwx
        b -= learning_rate*db
        return wh, wx, b

    def train(self,X_train,y_train,learning_rate = 0.1,n_epoch = 5):
        losses = []
        num_examples = 0

        for epoch in range(n_epoch):
            for i in range(len(y_train)):
                self.sgd_step(X_train[i],y_train[i],learning_rate)
                num_examples+=1
            loss = self.loss(X_train,y_train)
            losses.append(loss)
            print('epoch{0}:loss{1}'.format(epoch+1,loss))
            '''
            if len(losses) > 1 and losses[-1] >losses[-2]:
                learning_rate *= 0.5
                print('decrease learning_rate to',learning_rate)'''


if __name__ == '__main__':
    X = np.random.randint(0,10,(100,100))
    y = np.random.randint(0,2,(100,10))
    #print(X)
    print(y)
    lstm = toyLSTM(100,10)
    lstm.train(X,y,learning_rate = 0.01,n_epoch=5)
    output = lstm.predict(X[0])
    print(output)
