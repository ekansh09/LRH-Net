

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values



# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score



#F1score
def cal_Acc(y_true, y_pre, threshold=0.5, num_classes=9, beta=2, normal=False):
    
    y_true = y_true.cpu().detach().numpy().astype(np.int)

    y_label = np.zeros(y_true.shape)
    # Generate the one hot encoding labels
    _, y_pre_label = torch.max(y_pre, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(y_true.shape[0]), y_pre_label] = 1
    y_prob = y_pre.cpu().detach().numpy()
    y_pre = y_pre.cpu().detach().numpy() >= threshold


    y_label = y_label + y_pre
    y_label[y_label > 1.1] = 1

    labels = y_true
    binary_outputs = y_label


    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = '/scratch/physionet-2020/weights.csv'
    normal_class = '426783006'

    # Get the label
    label_file_dir = '/scratch/physionet-2020/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))
    

    weights = load_weights(weights_file, classes)

    # Only consider classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0) # Find indices of classes in weight matrix.
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]

    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)

    # Return the results.
    return challenge_metric

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    
    num_recordings, num_classes = np.shape(labels)
    outputs = outputs.cpu().detach().numpy().astype(np.int)
    labels = labels.cpu().detach().numpy().astype(np.int)
    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        #print("Sensitivity(tpr):",tpr," Specificity:",1-tnr)
        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    #print("auroc shape:",auroc.shape,"\nauroc:",auroc) #Class-wise aucroc
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return auroc, auprc


def print_confusion_matrix(confusion_matrix, axes, class_label,class_count, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - {} [{}]".format(class_label,class_count))


def plot_graphs(history,epochs):

    epoch_list = [i for i in range(epochs)]
    logs = list(history.keys())
    fig = plt.figure(figsize=(10, 10))
    rows = 2
    columns = 2
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .50)
    for i in range(rows*columns):
        #exec (f"plt.subplot(grid{[i]})")
        if i <=5:
            plt.plot(epoch_list, history[logs[i]])
            plt.title(logs[i])
    plt.show()



def get_predictions(labels,predictions,threshold):
    
    # Generate the outputs using the threshold
    labels_numpy = labels.cpu().detach().numpy().astype(np.int)
    
    y_label = np.zeros(labels_numpy.shape) #creating a dummy array

    _, y_pre_label = torch.max(predictions, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(labels_numpy.shape[0]), y_pre_label] = 1
    y_prob = predictions.cpu().detach().numpy()
    y_pre = predictions.cpu().detach().numpy() >= threshold

    y_predictions = y_pre + y_label
    y_predictions[y_predictions > 1.1] = 1
    
    return y_predictions


def visulaise_performance(ground_truth,predictions,history,epochs,threshold,class_names,classwise_sample_count):
    
    
    challenge_metric = cal_Acc(ground_truth, predictions, threshold, num_classes=len(class_names))
    print('Final Challenge Metric Score',challenge_metric)
    
    #plot_graphs(history,epochs)
    auroc,auprc = compute_auc(ground_truth, predictions)
    
    #fig = go.Figure(data=[go.Table(header=dict(values=['classes','AUROC', 'AUPRC']),
    #             cells=dict(values=[class_names, auroc,auprc]))
    #                 ])
    #fig.show()

    
    preds = get_predictions(ground_truth,predictions,threshold)
    
    report = skm.classification_report(ground_truth.cpu().detach().numpy(),preds, target_names = class_names)
    
    print(report)
    
    cm = skm.multilabel_confusion_matrix(ground_truth.cpu().detach().numpy(), preds)
    
    print(cm)
    
    #fig, ax = plt.subplots(6, 4, figsize=(22, 10))
    
    #for axes, cfs_matrix, label, class_count in zip(ax.flatten(), cm, class_names, classwise_sample_count):
    #    print_confusion_matrix(cfs_matrix, axes, label,class_count, ["N", "Y"])

    #fig.tight_layout()
    #plt.show()
