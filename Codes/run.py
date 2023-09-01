from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve




def train(criterion,optimizer,model,epoch,dataloader):
    
    model.train()
    
    # Define the temp variable
    epoch_start = time.time()
    epoch_loss = 0.0

    for batch_idx, (inputs, labels, age_gender,input_indices) in (enumerate(dataloader)):
        

        inputs = inputs.to(device)
        labels = labels.to(device)
        age_gender = age_gender.to(device)
        
        # Do the learning process, in val, we do not care about the gradient for relaxing
        with torch.set_grad_enabled(True):
            

            logits = model(inputs, age_gender) #check outputs
            sigmoid = nn.Sigmoid()
            logits_prob = sigmoid(logits)
            #print(logits_prob.shape)

            #saving the predictions and label
            if batch_idx == 0:
                labels_all = labels
                predictions_all = logits_prob
            else:
                labels_all = torch.cat((labels_all, labels), 0)
                predictions_all = torch.cat((predictions_all, logits_prob), 0)

            if 'Focal' in str(criterion):
                loss = criterion(logits,labels,dataloader.dataset.classwise_sample_count,len(dataloader.dataset))
            else:
                loss = criterion(logits,labels)
                
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

  
            
    train_auprc = average_precision_score(y_true=labels_all.cpu().detach().numpy(), y_score=predictions_all.cpu().detach().numpy())
    
    # Print the train and val information via each epoch
    epoch_loss = epoch_loss / len(dataloader)
    print('Training: Epoch: {} train-Loss: {:.4f} train-auprc {} Cost {:.1f} sec'.format(epoch, epoch_loss, train_auprc, time.time() - epoch_start))
    
    return epoch_loss,train_auprc

def validation(criterion, model, epoch, valid_dataloader,batch_size=16):
    
        
        model.eval()
        epoch_loss = 0
        dummy_input = torch.randn(16, 12, 5000, dtype=torch.float).to(device)
        age_dummy_input = torch.randn(16, 5, dtype=torch.float).to(device)
        total_time  = 0
        
        #GPU-WARM-UP: will automatically initialize the GPU and prevent it from going into power-saving mode when we measure time
        for _ in range(10):
            _ = model(dummy_input,age_dummy_input)

        
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            for batch_idx,(inputs ,labels,age_gender, _) in enumerate(valid_dataloader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                age_gender = age_gender.to(device)
                
                starter.record()
                logits = model(inputs,age_gender)
                sigmoid = nn.Sigmoid()
                logits_prob = sigmoid(logits)
                ender.record()
                
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000 #Returns time elapsed in milliseconds
                total_time += curr_time
                
                if 'Focal' in str(criterion):
                    loss = criterion(logits,labels,valid_dataloader.dataset.classwise_sample_count,len(valid_dataloader.dataset))
                else:
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
        

        
        valid_auprc = average_precision_score(y_true=labels_all.cpu().detach().numpy(), y_score=predictions_all.cpu().detach().numpy())

        print("Validation: loss:",epoch_loss," valid_auprc:",valid_auprc," total_time: ",total_time," secs")
        
        
        
        return labels_all,predictions_all,epoch_loss,valid_auprc
