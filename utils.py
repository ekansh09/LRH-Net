


# Define the learning rate decay

def get_optimizer(opt,model,lr,momentum,weight_decay):
    
    if opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                   momentum=momentum, weight_decay=weight_decay,nestrov = True)
    elif opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                    weight_decay=weight_decay)
    else:
 
        raise Exception("optimizer not implement")
    
    return optimizer

# Define the learning rate decay
def get_lr_scheduler(lr_scheduler,optimizer,steps,gamma):
    
    if lr_scheduler == 'step':
        steps_list = [int(step) for step in steps.split(',')]
        #print(steps_list)
        lr_scheduler_fn = optim.lr_scheduler.MultiStepLR(optimizer, steps_list, gamma=gamma)
    elif lr_scheduler == 'exp':
        lr_scheduler_fn = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif lr_scheduler == 'stepLR':
        steps = int(steps)
        lr_scheduler_fn = optim.lr_scheduler.StepLR(optimizer, steps, gamma)
    elif lr_scheduler == 'cos':
        steps = int(steps)
        lr_scheduler_fn = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
    elif lr_scheduler == 'fix':
        lr_scheduler_fn = None
    else:
        raise Exception("lr schedule not implement")
        
    return lr_scheduler_fn

# Define the monitoring accuracy
def accuracy_measuring_fn(monitor_acc):
    
    if monitor_acc == 'acc':
        cal_acc = None
    elif monitor_acc == 'AUC':
        cal_acc = RocAucEvaluation
    elif monitor_acc == 'ecgAcc':
        cal_acc = cal_Acc
    else:
        raise Exception("monitor_acc is not implement")
        
    return cal_acc


class optim_genetics:
    def __init__(self, target, outputs, classes):
        self.target = target
        self.outputs = outputs
        weights_file = '/scratch/physionet-2020/weights.csv'
        self.normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'],
                              ['284470004', '63593006'],
                              ['427172004', '17338001']]

        # Load the scored classes and the weights for the Challenge metric.
        self.weights = load_weights(weights_file, classes)
        self.classes = classes
        # match classed ordering
        # reorder = [self.classes.index(c) for c in classes]
        # self.outputs = self.outputs[:, reorder]
        # self.target = self.target[:, reorder]
        stop = 1

    def __call__(self, x):
        outputs = copy.deepcopy(self.outputs)
        outputs = outputs > x
        outputs = np.array(outputs, dtype=int)
        return -compute_challenge_metric(self.weights, self.target, outputs, self.classes, self.normal_class)



def find_thresholds(t,y, class_codes):

    N = 24
    f1prcT = np.zeros((N,))
    f1rocT = np.zeros((N,))

    for j in range(N):
        prc, rec, thr = precision_recall_curve(y_true=t[:, j], probas_pred=y[:, j])
        fscore = 2 * prc * rec / (prc + rec)
        idx = np.nanargmax(fscore)
        f1prc = np.nanmax(fscore)
        f1prcT[j] = thr[idx]

        fpr, tpr, thr = roc_curve(y_true=t[:, j], y_score=y[:, j])
        fscore = 2 * (1 - fpr) * tpr / (1 - fpr + tpr)
        idx = np.nanargmax(fscore)
        f1roc = np.nanmax(fscore)
        f1rocT[j] = thr[idx]

    population = np.random.rand(300, N)
    for i in range(1, 99):
        population[i, :] = i / 100

#     print(f1prcT)
#     print(f1rocT)
    population[100] = f1rocT
    population[101] = f1prcT
    bounds = [(0, 1) for i in range(N)]
    print('Differential_evolution started...')
    result = differential_evolution(optim_genetics(t, y, class_codes), bounds=bounds, disp=True, init=population, workers=-1)
    print('Differential_evolution ended...')
#     print(result)
    return result.x
