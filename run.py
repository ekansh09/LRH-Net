from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve




def train(criterion,optimizer,model,epoch,dataloader,cal_acc, threshold,scaler):
    
    model.train()
    
    # Define the temp variable
    epoch_start = time.time()
    epoch_loss = 0.0
    
    epoch_start = time.time()

    for batch_idx, (inputs, labels, input_indices) in (enumerate(dataloader)):

        inputs = inputs.to(device)
        labels = labels.to(device)
            
        with amp.autocast():

            logits = model(inputs) #check outputs
            sigmoid = nn.Sigmoid()
            logits_prob = sigmoid(logits)

            # output is float16 because linear layers autocast to float16.
            assert logits_prob.dtype is torch.float16

            #saving the predictions and label
            if batch_idx == 0:
                labels_all = labels
                predictions_all = logits_prob
            else:
                labels_all = torch.cat((labels_all, labels), 0)
                predictions_all = torch.cat((predictions_all, logits_prob), 0)

            loss = criterion(logits,labels)
            # loss is float32 because mse_loss layers autocast to float32.
            assert loss.dtype is torch.float32

            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp
            
            if torch.isnan(torch.tensor(loss_temp)):
                print("batch_nan:",batch_idx,"loss_batch:", loss_temp,"input:", torch.isnan(inputs).sum(), torch.isnan(logits).sum()  )

            if batch_idx%300 == 0:
                print("batch__:",batch_idx,"loss:", epoch_loss )
                
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()
        
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        
        # Updates the scale for next iteration.
        scaler.update()
        
        # set_to_none=True here can modestly improve performance
        optimizer.zero_grad() 

            
    challenge_metric = cal_acc(labels_all, predictions_all, threshold, num_classes=len(dataloader.dataset.class_names))
    
    print('before',epoch_loss)
    
    # Print the train and val information via each epoch
    epoch_loss = epoch_loss / len(dataloader)
    
    print('after',epoch_loss)
    
    logging.info('Epoch: {} train-Loss: {:.4f} train-challenge_metric {} Cost {:.1f} sec'.format(epoch, epoch_loss,challenge_metric, time.time() - epoch_start))
    
    print('Training: Epoch: {} train-Loss: {:.4f} train-challenge_metric {} Cost {:.1f} sec'.format(epoch, epoch_loss, challenge_metric, time.time() - epoch_start))
    
    return epoch_loss,challenge_metric



def validation(criterion, model, epoch, valid_dataloader,threshold,cal_acc,batch_size=batch_size):
    
        
        model.eval()
        epoch_loss = 0
        dummy_input = torch.randn(batch_size, 12, 5000, dtype=torch.float).to(device)
        total_time  = 0
        
        #GPU-WARM-UP: will automatically initialize the GPU and prevent it from going into power-saving mode when we measure time
        for _ in range(10):
            _ = model(dummy_input)

        
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            for batch_idx,(inputs ,labels, _) in enumerate(valid_dataloader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                starter.record()
                logits = model(inputs)
                sigmoid = nn.Sigmoid()
                logits_prob = sigmoid(logits)
                ender.record()
                
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000 #Returns time elapsed in milliseconds
                total_time += curr_time
                
                loss = criterion(logits,labels)
                loss_temp = loss.item() * inputs.size(0)
                epoch_loss += loss_temp
                

                
                #storing all the predictions
                if batch_idx == 0:  
                    labels_all = labels
                    predictions_all = logits_prob
                else:
                    labels_all = torch.cat((labels_all,labels), 0)
                    predictions_all = torch.cat((predictions_all, logits_prob), 0)
                
                    
        epoch_loss = epoch_loss / len(valid_dataloader)
        
        #This formula gives the number of examples our network can process in one second.
        Throughput = (batch_size)/total_time
        

        challenge_metric = cal_acc(labels_all, predictions_all, threshold, num_classes=len(valid_dataloader.dataset.class_names))
        auroc,auprc = compute_auc(labels_all, predictions_all)
        print("Validation: loss:",epoch_loss," metric_value:",challenge_metric," total_time: ",total_time," secs")
        
        return labels_all,predictions_all,epoch_loss,challenge_metric
