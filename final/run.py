from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve




def train(criterion,divergence_loss_fn,optimizer,student_model,teacher_model1,teacher_model2,epoch,dataloader,cal_acc, threshold,alpha,temp,leads,selected_leads,scaler):
    
    student_model.train()
    teacher_model1.eval()
    teacher_model2.eval()
    # Define the temp variable
    epoch_start = time.time()
    epoch_loss = 0.0
    distillation_loss = 0.0
    

    for batch_idx, (inputs, labels, age_gender,input_indices) in (enumerate(dataloader)):
        
        if selected_leads != None:
            lead_pos = [leads[i] for i in selected_leads]
            student_inputs = inputs[:,lead_pos,:]
        else:
            student_inputs = inputs
        teacher_inputs = inputs.to(device)
        student_inputs = student_inputs.to(device)
        labels = labels.to(device)
        age_gender = age_gender.to(device)
        
        with torch.no_grad():
            teacher1_logits = teacher_model1(teacher_inputs,age_gender)
            teacher2_logits = teacher_model2(student_inputs,age_gender)
        
        # Do the learning process, in val, we do not care about the gradient for relaxing
        with torch.set_grad_enabled(True):
            
            student_logits = student_model(student_inputs, age_gender) #check outputs
            sigmoid = nn.Sigmoid()
            logits_prob = sigmoid(student_logits)
            #print(logits_prob.shape)

            #saving the predictions and label
            if batch_idx == 0:
                labels_all = labels
                predictions_all = logits_prob
            else:
                labels_all = torch.cat((labels_all, labels), 0)
                predictions_all = torch.cat((predictions_all, logits_prob), 0)

            if 'Focal' in str(criterion):
                student_loss = criterion(student_logits,labels,dataloader.dataset.classwise_sample_count,len(dataloader.dataset))
            else:
                student_loss = criterion(student_logits,labels)
                            
            ditillation_loss1 = divergence_loss_fn(   
                F.log_softmax((student_logits / temp), dim = 1),
                F.softmax((teacher1_logits / temp), dim = 1))
            
            ditillation_loss2 = divergence_loss_fn(   
                F.log_softmax((student_logits / temp), dim = 1),
                F.softmax((teacher2_logits / temp), dim = 1))
            
            
            loss = alpha * student_loss + (1 - alpha) *0.70* ditillation_loss1 + (1 - alpha) *.30* ditillation_loss2 

                
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp
            
            distill_loss = (ditillation_loss1.item()+ditillation_loss2.item()) * inputs.size(0)
            distillation_loss += distill_loss
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

  
    challenge_metric = cal_acc(labels_all, predictions_all, threshold, num_classes=len(dataloader.dataset.class_names))
    
    # Print the train and val information via each epoch
    epoch_loss = epoch_loss / len(dataloader)
    distillation_loss = distillation_loss / len(dataloader)
    
    logging.info('Epoch: {} train-Loss: {:.4f} train-challenge_metric {} Cost {:.1f} sec'.format(epoch, epoch_loss,challenge_metric, time.time() - epoch_start))
    
    print('Training: Epoch: {} train-Loss: {:.4f} distillation-Loss: {:.4f} train-challenge_metric {} Cost {:.1f} sec'.format(epoch, epoch_loss,distillation_loss, challenge_metric, time.time() - epoch_start))
    
    return epoch_loss,challenge_metric, distillation_loss



def validation(criterion, model, epoch, valid_dataloader,threshold,leads,selected_leads,cal_acc,batch_size=batch_size):
    
        model.eval()
        epoch_loss = 0
        inp_channel = 12 if selected_leads == None else len(selected_leads)
        dummy_input = torch.randn(16,inp_channel, 5000, dtype=torch.float).to(device)
        age_dummy_input = torch.randn(16, 5, dtype=torch.float).to(device)
        total_time  = 0
        
        #GPU-WARM-UP: will automatically initialize the GPU and prevent it from going into power-saving mode when we measure time
        for _ in range(10):
            _ = model(dummy_input,age_dummy_input)

        
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            for batch_idx,(inputs ,labels,age_gender, _) in enumerate(valid_dataloader):
                
                
                if selected_leads != None:
                    lead_pos = [leads[i] for i in selected_leads]
                    inputs = inputs[:,lead_pos,:]
                else:
                    inputs = inputs

                
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
        

        challenge_metric = cal_acc(labels_all, predictions_all, threshold, num_classes=len(valid_dataloader.dataset.class_names))
        auroc,auprc = compute_auc(labels_all, predictions_all)
        print("Validation: loss:",epoch_loss," metric_value:",challenge_metric," total_time: ",total_time," secs")
        
        
        
        return labels_all,predictions_all,epoch_loss,challenge_metric
