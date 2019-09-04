import os
import torch
import copy
import math
import numpy as np
import itertools
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
from src.asr import Seq2Seq
from src.rnnlm import RNN_LM
from src.clm import CLM_wrapper
from src.dataset import LoadDataset
from src.postprocess import Mapper,cal_acc,cal_cer,draw_att
from src.acoustic_classifier_networks import LSTMClassifier, AttentionModel, SelfAttention

import logging

VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 250 # steps for debugging info.
GRAD_CLIP = 5
CLM_MIN_SEQ_LEN = 5

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()[:,1]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        logging.warning("all values are equal!")
        auc = 0.5
    return auc

class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config,paras):
        # General Settings
        self.config = config
        self.paras = paras
        gpu_no = str(paras.gpu_no)
        self.device = torch.device('cuda:' + gpu_no) if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        if not os.path.exists(paras.ckpdir):os.makedirs(paras.ckpdir)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        if not os.path.exists(self.ckpdir):os.makedirs(self.ckpdir)
        
        # Load Mapper for idx2token
        self.mapper = Mapper(config['solver']['data_path'])

        self.asr_model_file = paras.asr_model_file
        self.acoustic_classifier_model_file = paras.acoustic_classifier_model_file

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[INFO]',msg)
   
    def progress(self,msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose:
            print(msg+'                              ',end='\r')


class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self,config,paras):
        super(Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_val_ed = 2.0
        self.best_ac_acoustic_val_auc = 0.0

        # Training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.tf_start = config['solver']['tf_start']
        self.tf_end = config['solver']['tf_end']
        self.apex = config['solver']['apex']

        # CLM option
        self.apply_clm = config['clm']['enable']

    def load_data(self):
        ''' Load date for training/validation'''
        self.verbose('Loading data from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
       
        # Get 1 example for auto constructing model
        for self.sample_x,_,_,_ in getattr(self,'train_set'):break
        if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

    def set_model(self):
        ''' Setup ASR (and CLM if enabled)'''
        self.verbose('Init ASR model. Note: validation is done through greedy decoding w/ attention decoder.')
        
        if self.config["acoustic_classification"]["model_type"] == "LSTM":
            self.acoustic_classifier = LSTMClassifier(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        elif self.config["acoustic_classification"]["model_type"] == "ATTENTION-LSTM":
            self.acoustic_classifier = AttentionModel(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        elif self.config["acoustic_classification"]["model_type"] == "SelfAttention":
            self.acoustic_classifier = SelfAttention(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        else:
            self.verbose("Error: AC model type is not known") 


        # Build attention end-to-end ASR
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
        if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
            self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
        
        # Involve CTC
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']
        
        self.classification_weight = self.config['asr_model']['optimizer']['classification_weight']
            
        # Setup optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])   
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)
        
        
        if self.apex and self.config['acoustic_classification']['optimizer']['type']=='Adam':
            import apex 
            self.ac_classifier_opt = apex.optimizers.FusedAdam(self.acoustic_classifier.parameters(), lr=self.config['acoustic_classification']['optimizer']['learning_rate'])
        else:
            self.ac_classifier_opt = getattr(torch.optim,self.config['acoustic_classification']['optimizer']['type'])
            self.ac_classifier_opt = self.ac_classifier_opt(self.acoustic_classifier.parameters(), lr=self.config['acoustic_classification']['optimizer']['learning_rate'],eps=1e-8)
        

        if self.paras.load:
            
            start_over = False
            if  self.acoustic_classifier_model_file is None or self.asr_model_file is None:
                self.step = 0  #TODO is this correct?
                self.verbose("step set to 0.") 
                start_over = True

            #checkpoint = torch.load(os.path.join(self.ckpdir,'asr'), map_location=self.device)
            if  self.asr_model_file is not None:
                checkpoint = torch.load(self.asr_model_file, map_location=self.device)
                self.asr_model.load_state_dict(checkpoint['model_state_dict'])
                self.asr_model.to(self.device)
                if start_over is True:
                    checkpoint['optimizer_state_dict']["step"] = self.step
                else:
                    self.step = checkpoint['step']

                self.asr_opt.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.asr_loss = checkpoint['asr_loss']
                self.asr_model.train()
                self.verbose("ASR model is loaded!")
            else:
                self.verbose("ASR model trains from sctrach!")
        
            #checkpoint = torch.load(os.path.join(self.ckpdir,'acoustic_classifier'), map_location=self.device)
            if self.acoustic_classifier_model_file is not None:
                checkpoint = torch.load(self.acoustic_classifier_model_file, map_location=self.device)
                self.acoustic_classifier.load_state_dict(checkpoint['model_state_dict'])
                self.acoustic_classifier.to(self.device)
                self.ac_classifier_opt.load_state_dict(checkpoint['optimizer_state_dict'])
                if start_over is True:
                    checkpoint['optimizer_state_dict']["step"] = self.step
                else:
                    self.step = checkpoint['step']

                #if checkpoint['step'] > self.step:
                #    self.step = checkpoint['step']  # TODO whcih step? 
                self.ac_classification_loss = checkpoint['ac_classification_loss']
                self.acoustic_classifier.train()
                self.verbose("Acoustic classifier is loaded")
            else: 
                self.verbose("Acoustic classifier trains from scratch")

            #if  self.acoustic_classifier_model_file is None or self.asr_model_file is None:
            #    self.step = 0
            #    self.verbose("step set to 0.")  # TODO  not working: switch back to saved number and back again print garbage in log

            self.verbose("pretrained models are loaded.")
            

        # Apply CLM
        if self.apply_clm:
            self.clm = CLM_wrapper(self.mapper.get_dim(), self.config['clm']).to(self.device)
            clm_data_config = self.config['solver']
            clm_data_config['train_set'] = self.config['clm']['source']
            clm_data_config['use_gpu'] = self.paras.gpu
            self.clm.load_text(clm_data_config)
            self.verbose('CLM is enabled with text-only source: '+str(clm_data_config['train_set']))
            self.verbose('Extra text set total '+str(len(self.clm.train_set))+' batches.')


        # freeze models 
        if self.config["full_model"]["asr_decoder_freeze"] is True:
            self.verbose("freeze asr (encoder/decoder).")
            for p in self.asr_model.parameters():
                p.requires_grad = False
        if self.config["full_model"]["asr_encoder_freeze"] is False:       
            self.verbose("ubfreeze asr encoder.") 
            for p in self.asr_model.encoder.parameters():
                p.requires_grad = True
        if self.config["full_model"]["acoustic_classifier_freeze"] is True: 
            self.verbose("freeze acoustic classifier")
            for p in self.acoustic_classifier.parameters():
                p.requires_grad = False

    def exec(self):
        ''' Training End-to-end ASR system'''
        self.verbose('Training set total '+str(len(self.train_set))+' batches.')

        while self.step < self.max_step:
            for x, y, z, fname in self.train_set:
           
                self.progress('Training step - '+str(self.step))
                # Perform teacher forcing rate decaying
                tf_rate = self.tf_start - self.step*(self.tf_start-self.tf_end)/self.max_step
                
                # Hack bucket, record state length for each uttr, get longest label seq for decode step
                assert len(x.shape)==4,'Bucketing should cause acoustic feature to have shape 1xBxTxD'
                assert len(y.shape)==3,'Bucketing should cause label have to shape 1xBxT'
                x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                z = torch.squeeze(torch.stack(z)).long().to(self.device) #z.squeeze(0).to(device = self.device,dtype=torch.long)
                state_len = np.sum(np.sum(x.cpu().data.numpy(),axis=-1)!=0,axis=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # ASR forwarding 
                self.asr_opt.zero_grad()
                self.ac_classifier_opt.zero_grad()

                ctc_pred, state_len, att_pred, _,encode_feature, vgg_feature, vgg_enc_len =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)
                

                # Acoustic classifer forwarding and loss
                # TODO ?  is it correct 
                # https://discuss.pytorch.org/t/implement-multi-input-multi-head-neural-network-with-different-specific-forward-backpropagation-path/18360
                # https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/14
                
                if self.config["acoustic_classification"]["input"] == "encoder":
                    temp = encode_feature
                    temp_d = temp.detach()
                    temp_d.requires_grad = True
                elif self.config["acoustic_classification"]["input"] == "VGG":
                    temp = vgg_feature
                    temp_d = temp.detach()
                    temp_d.requires_grad = True
                else: 
                    self.verbose('Error: acoustic input is not know')


                logits, class_pred = self.acoustic_classifier(temp_d, temp_d.shape[0])

                self.ac_classification_loss = torch.nn.CrossEntropyLoss()(logits, z)
               
                loss_log = {}
                label = y[:,1:ans_len+1].contiguous()
                ctc_loss = 0
                att_loss = 0

                loss_log["ac_classification_loss"] = self.ac_classification_loss

                # CE loss on attention decoder
                if self.ctc_weight<1:
                    b,t,c = att_pred.shape
                    att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
                    att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                               .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                    att_loss = torch.mean(att_loss) # Mean by batch
                    loss_log['train_att'] = att_loss

                # CTC loss on CTC decoder
                if self.ctc_weight>0:
                    target_len = torch.sum(y!=0,dim=-1)
                    ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                    loss_log['train_ctc'] = ctc_loss
                
                # ASR loss is composed of both ASR loss and classifiers loss  (with weights)
                # This allows: 1- learn the asr in a way useful for classification tasks.  2- finetune the asr or encoder in later steps.
                self.asr_loss = (1- self.classification_weight)*((1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss) +   \
                                         self.classification_weight * self.ac_classification_loss  # TODO add other weights

                loss_log['train_full'] = self.asr_loss
                
                # Adversarial loss from CLM
                if self.apply_clm and att_pred.shape[1]>=CLM_MIN_SEQ_LEN:
                    if (self.step%self.clm.update_freq)==0:
                        # update CLM once in a while
                        clm_log,gp = self.clm.train(att_pred.detach(),CLM_MIN_SEQ_LEN)
                        self.write_log('clm_score',clm_log)
                        self.write_log('clm_gp',gp)
                    adv_feedback = self.clm.compute_loss(F.softmax(att_pred))
                    self.asr_loss -= adv_feedback

                # Backprop
                # asr
                #self.asr_loss.backward(retain_graph=True)
                #loss = self.asr_loss + self.ac_classification_loss
                #loss.backward()
                ## TODO aobe  works?  or:
                self.ac_classification_loss.backward(retain_graph=True)
                # TODO ? why not above?
                #temp.backward(torch.autograd.grad(self.ac_classification_loss, temp_d, only_inputs=False, retain_graph=True)[0], retain_graph=True)
                self.asr_loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step (asr)'+str(self.step))
                else:
                    self.asr_opt.step()
                
                # acoustic classifier
                #self.ac_classification_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.acoustic_classifier.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step (ac_classifier)'+str(self.step))
                else:
                    self.ac_classifier_opt.step()

                # Logger
                self.write_log('loss',loss_log)
                if self.ctc_weight<1:
                    self.write_log('acc',{'train':cal_acc(att_pred,label)})
                if self.step % TRAIN_WER_STEP ==0:
                    self.write_log('error rate',
                                   {'train':cal_cer(att_pred,label,mapper=self.mapper)})

                # Validation
                if self.step%self.valid_step == 0:
                    self.asr_opt.zero_grad()
                    self.ac_classifier_opt.zero_grad()
                    self.valid()

                self.step+=1
                if self.step > self.max_step:break
    

    def write_log(self,val_name,val_dict):
        '''Write log to TensorBoard'''
        if 'att' in val_name:
            self.log.add_image(val_name,val_dict,self.step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, self.step)
        else:
            self.log.add_scalars(val_name,val_dict,self.step)


    def valid(self):
    
        '''Perform validation step (!!!NOTE!!! greedy decoding with Attention decoder only)'''
        self.asr_model.eval()
        self.acoustic_classifier.eval()

        # Init stats
        val_loss, val_ctc, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        
        total_acc = 0.0
        total_auc = 0.0
        val_ac_classification_loss = 0.0
        # Perform va idation
        for cur_b,(x,y,z, fname) in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.dev_set)),')']))

            # Prepare data
            if len(x.shape)==4: x = x.squeeze(0)
            if len(y.shape)==3: y = y.squeeze(0)
            x = x.to(device = self.device,dtype=torch.float32)
            y = y.to(device = self.device,dtype=torch.long)
            z = torch.squeeze(torch.stack(z)).long().to(self.device) 
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            ctc_pred, state_len, att_pred, att_maps, encode_feature, vgg_feature, vgg_enc_len  = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Acoustic classifer forwarding and loss
            # TODO ?  is it correct 
            # https://discuss.pytorch.org/t/implement-multi-input-multi-head-neural-network-with-different-specific-forward-backpropagation-path/18360
            # https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/14
            #temp = encode_feature
            #temp_d = temp.detach()
            #temp_d.requires_grad = True
            if self.config["acoustic_classification"]["input"] == "encoder":
                temp = encode_feature
                temp_d = temp.detach()
                temp_d.requires_grad = True
            elif self.config["acoustic_classification"]["input"] == "VGG":
                temp = vgg_feature
                temp_d = temp.detach()
                temp_d.requires_grad = True
            else: 
                self.verbose('Error: acoustic input is not know')

            logits, class_pred = self.acoustic_classifier(temp_d, temp_d.shape[0])
            ##logits, class_pred = self.acoustic_classifier(encode_feature, encode_feature.shape[0])
            val_ac_classification_loss += torch.nn.CrossEntropyLoss()(logits, z) * int(x.shape[0])

            target = z #torch.squeeze(torch.stack(z)).long()
            num_corrects = (torch.max(class_pred, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects
            auc = roc_auc_compute_fn(class_pred.cpu().detach(), target.cpu())
            #total_epoch_loss += loss.item()
            total_acc += acc.item()
            total_auc += auc * int(x.shape[0])

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
            if self.ctc_weight<1:
                seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
                seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1)\
                           .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                seq_loss = torch.mean(seq_loss) # Mean by batch
                val_att += seq_loss.detach()*int(x.shape[0])
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_acc += cal_acc(att_pred,label)*int(x.shape[0])
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
            
            # Compute CTC loss
            if self.ctc_weight>0:
                target_len = torch.sum(y!=0,dim=-1)
                ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, 
                                         torch.LongTensor(state_len), target_len)
                val_ctc += ctc_loss.detach()*int(x.shape[0])

            val_len += int(x.shape[0])
        

        avg_acc = total_acc / val_len
        avg_auc = total_auc / val_len
        
        acc_log = {}
        auc_log = {}
        for k,v in zip(["dev_ac_classification"], [avg_acc]):
            if v > 0.0: acc_log[k] = v
        self.write_log('acuracy',acc_log)

        for k,v in zip(["dev_ac_classification"], [avg_auc]):
            if v > 0.0: auc_log[k] = v
        self.write_log('AUC', auc_log)

        # Logger
        val_loss = (1 - self.classification_weight) * ((1-self.ctc_weight)*val_att + self.ctc_weight*val_ctc)+ \
                    self.classification_weight * val_ac_classification_loss  # TODO:  add  char loss

        loss_log = {}
        loss_log["dev_ac_classification"] = val_ac_classification_loss / val_len
        for k,v in zip(['dev_full','dev_ctc','dev_att'],[val_loss, val_ctc, val_att]):
            if v > 0.0: loss_log[k] = v/val_len
        self.write_log('loss',loss_log)
 
        if self.ctc_weight<1:
            # Plot attention map to log
            val_hyp,val_txt = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
            val_attmap = draw_att(att_maps,att_pred)

            # Record loss
            self.write_log('error rate',{'dev':val_cer/val_len})
            self.write_log('acc',{'dev':val_acc/val_len})
            for idx,attmap in enumerate(val_attmap):
                self.write_log('att_'+str(idx),attmap)
                self.write_log('hyp_'+str(idx),val_hyp[idx])
                self.write_log('txt_'+str(idx),val_txt[idx])

            # Save model by val er.
            if val_cer/val_len  < self.best_val_ed:
                self.best_val_ed = val_cer/val_len
                self.verbose('Best val er       : {:.4f}       @ step {}'.format(self.best_val_ed,self.step))
                #torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
                torch.save({
                'step': self.step,
                'model_state_dict': self.asr_model.state_dict(),
                'optimizer_state_dict': self.asr_opt.state_dict(),
                'asr_loss': self.asr_loss,
                }, os.path.join(self.ckpdir,'asr_asrpoint'))
                
                torch.save({
                'step': self.step,
                'model_state_dict': self.acoustic_classifier.state_dict(),
                'optimizer_state_dict': self.ac_classifier_opt.state_dict(),
                'ac_classification_loss': self.ac_classification_loss,
                }, os.path.join(self.ckpdir,'acoustic_classifier_asrpoint'))
                
                if self.apply_clm:
                    torch.save(self.clm.clm,  os.path.join(self.ckpdir,'clm'))
                # Save hyps.
                with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                    for t1,t2 in zip(all_pred,all_true):
                        f.write(t1+','+t2+'\n')

            # save model by auc
            if avg_auc > self.best_ac_acoustic_val_auc:
                self.best_ac_acoustic_val_auc = avg_auc
                self.verbose('Best val auc       : {:.4f}       @ step {}'.format(self.best_ac_acoustic_val_auc, self.step))
                # TODO  is this better or the above?
                torch.save({
                'step': self.step,
                'model_state_dict': self.asr_model.state_dict(),
                'optimizer_state_dict': self.asr_opt.state_dict(),
                'asr_loss': self.asr_loss,
                }, os.path.join(self.ckpdir,'asr_acpoint'))

                torch.save({
                'step': self.step,
                'model_state_dict': self.acoustic_classifier.state_dict(),
                'optimizer_state_dict': self.ac_classifier_opt.state_dict(),
                'ac_classification_loss': self.ac_classification_loss,
                }, os.path.join(self.ckpdir,'acoustic_classifier_acpoint'))

        self.asr_model.train()
        self.acoustic_classifier.train()

class Validator(Solver):
    """
        Prediction mode for classifiers.
        TODO: 
            1- support mode with no labels (Prediction)
            2- support online Prediction

    """

    def __init__(self,config,paras):
        super(Validator, self).__init__(config, paras)
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        

    def load_data(self):
        self.verbose('Loading data for validation'+' from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'test_set',LoadDataset('test',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        
        # TODO: can we load the model simply?
        # Get 1 example for auto constructing model
        for self.sample_x,_,_,_ in getattr(self,'train_set'):break
        if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

    def set_model(self):
        ''' Load saved ASR'''
        print("***** ", self.device)
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        #self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))
        #checkpoint = torch.load(os.path.join(self.ckpdir,'asr'), map_location=self.device)
        checkpoint = torch.load(self.asr_model_file, map_location=self.device)
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
        self.asr_model.load_state_dict(checkpoint['model_state_dict'])
        self.asr_model.to(self.device)
        
        # Load Acoustic classifer
        self.verbose('Load acoustic classifier model from '+os.path.join(self.ckpdir))
        #checkpoint = torch.load(os.path.join(self.ckpdir,'acoustic_classifier'), map_location=self.device)
        checkpoint = torch.load(self.acoustic_classifier_model_file, map_location=self.device)
        
        if self.config["acoustic_classification"]["model_type"] == "LSTM":
            self.acoustic_classifier = LSTMClassifier(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        elif self.config["acoustic_classification"]["model_type"] == "ATTENTION-LSTM":
            self.acoustic_classifier = AttentionModel(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        elif self.config["acoustic_classification"]["model_type"] == "SelfAttention":
            self.acoustic_classifier = SelfAttention(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)
        else:
            self.verbose("Error: AC model type is not known")
        

        #self.acoustic_classifier = LSTMClassifier(self.sample_x, self.config['acoustic_classification'], self.config['asr_model']).to(self.device)  #TODO  move the params
        #self.acoustic_classifier = torch.load(os.path.join(self.ckpdir,'acoustic_classifier'))
        self.acoustic_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.acoustic_classifier.to(self.device)

        # Enable joint CTC decoding
        self.asr_model.joint_ctc = self.config['solver']['decode_ctc_weight'] >0
        if self.config['solver']['decode_ctc_weight'] >0:
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.verbose('Joint CTC decoding is enabled with weight = '+str(self.config['solver']['decode_ctc_weight']))
            #self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
            self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']

        
        # TODO: Can we still have RNN decoding?
        """    
        # Enable joint RNNLM decoding
        self.decode_lm = self.config['solver']['decode_lm_weight'] >0
        setattr(self.asr_model,'decode_lm_weight',self.config['solver']['decode_lm_weight'])
        if self.decode_lm:
            assert os.path.exists(self.config['solver']['decode_lm_path']), 'Please specify RNNLM path.'
            self.asr_model.load_lm(**self.config['solver'])
            self.verbose('Joint RNNLM decoding is enabled with weight = '+str(self.config['solver']['decode_lm_weight']))
            self.verbose('Loading RNNLM from '+self.config['solver']['decode_lm_path'])
            self.decode_file += '_lm{:}'.format(self.config['solver']['decode_lm_weight'])
        """


        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model.clear_att()
        self.asr_model = self.asr_model.to(self.device)
        ####self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        ####self.valid()
        ####self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

        self.acoustic_classifier.eval()
        self.acoustic_classifier = self.acoustic_classifier.to(self.device)

    #def exec(self):
    #    '''Perform inference step with beam search decoding.'''
    #    test_cer = 0.0
    #    self.decode_beam_size = self.config['solver']['decode_beam_size']
    #    self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
    #    self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
    #    ## self.test_set = [(x,y) for (x,y) in self.test_set][::10]
    #    _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
    #    
    #    self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
    #    
    #    self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
    #                                               str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
    


    def exec(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        
      
        with torch.no_grad():
            with open("XXXXX.out", "w") as fo:  #TODO  get output file from cmd or config
                fo.write("split,fname,score\n")
                for sub_set in ["self.train_set", "self.dev_set", "self.test_set"]:
                    val_cer = 0.0
                    val_len = 0    
                    all_pred,all_true = [],[]
                    ctc_results = []
                    total_acc  = 0.0
                    total_auc = 0.0
                    if sub_set == "self.train_set":
                        split = "train"
                    elif sub_set == "self.dev_set":
                        split = "dev"
                    elif sub_set == "self.test_set":
                        split = "test"
                    sub_set = eval(sub_set)
                    for cur_b,(x, y, z, fname) in enumerate(sub_set):
                        #self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.dev_set)),')']))

                        # Prepare data
                        if len(x.shape)==4: x = x.squeeze(0)
                        if len(y.shape)==3: y = y.squeeze(0)
                        x = x.to(device = self.device,dtype=torch.float32)
                        y = y.to(device = self.device,dtype=torch.long)
                        z = torch.squeeze(torch.stack(z)).long().to(self.device)
                        if len(fname) > 0:
                            fname = list(np.squeeze(fname))
                        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
                        state_len = [int(sl) for sl in state_len]
                        ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                        # Forward 
                        max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio)) # TODO is  it correct?
                        ctc_pred, state_len, att_pred, att_maps, encode_feature, vgg_feature, vgg_enc_len  = self.asr_model(x, max_decode_step,state_len=state_len)
                        ctc_pred = torch.argmax(ctc_pred,dim=-1).cpu() if ctc_pred is not None else None
                        ctc_results.append(ctc_pred)

                        # Acoustic classifer forwarding and loss
                        logits, class_pred = self.acoustic_classifier(encode_feature, encode_feature.shape[0])
                        #val_ac_classification_loss = torch.nn.CrossEntropyLoss()(logits, z)
                        target = z #torch.squeeze(torch.stack(z)).long()
                        num_corrects = (torch.max(class_pred, 1)[1].view(target.size()).data == target.data).sum()
                        acc = 100.0 * num_corrects
                        auc = roc_auc_compute_fn(class_pred.cpu().detach(), target.cpu())
                        #total_epoch_loss += loss.item()
                        total_acc += acc.item()
                        total_auc += auc * int(x.shape[0])

                        # Result
                        label = y[:,1:ans_len+1].contiguous()
                        t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                        all_pred += t1
                        all_true += t2
                        val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
                        val_len += int(x.shape[0])

                        # write fname and scores  (fname shouldbe mapped to ones we need later)
                        for fi,cp in zip(fname, class_pred.cpu().numpy()):
                            fo.write(split + "," + fi + "," + str(cp[1]) + "\n")
                           
                    avg_auc = total_auc / val_len
                    avg_acc = total_acc / val_len
                    # TODO : write to a report file
                    print("split:", split)
                    print("avg. accuracy", avg_acc)
                    print("avg. AUC", avg_auc)
                
                # TODO print reports
                # TODO save asr results in proper files          
  
            # TODO: add code to write results and reports
            """ 
            # Dump model score to ensure model is corrected
            self.verbose('Validation Error Rate of Current model : {:.4f}      '.format(val_cer/val_len)) 
            self.verbose('See {} for validation results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
                for hyp,gt in zip(all_pred,all_true):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
            
            # Also dump CTC result if available
            if ctc_results[0] is not None:
                ctc_results = [i for ins in ctc_results for i in ins]
                ctc_text = []
                for pred in ctc_results:
                    p = [i for i in pred.tolist() if i != 0]
                    p = [k for k, g in itertools.groupby(p)]
                    ctc_text.append(self.mapper.translate(p,return_string=True))
                self.verbose('Also, see {} for CTC validation results.'.format(os.path.join(self.ckpdir,'dev_ctc_decode.txt'))) 
                with open(os.path.join(self.ckpdir,'dev_ctc_decode.txt'),'w') as f:
                    for hyp,gt in zip(ctc_text,all_true):
                        f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
            """

# This is only for module using beam-search (slow). (classification is not implemented)
# predction class is for classification module and use greedy decoding (fast)
class Tester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self,config,paras):
        super(Tester, self).__init__(config,paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        
        self.decode_file = "_".join(['decode','beam',str(self.config['solver']['decode_beam_size']),
                                     'len',str(self.config['solver']['max_decode_step_ratio'])])

    def load_data(self):
        self.verbose('Loading testing data '+str(self.config['solver']['test_set'])\
                     +' from '+self.config['solver']['data_path'])
        setattr(self,'test_set',LoadDataset('test',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))
        
        # Enable joint CTC decoding
        self.asr_model.joint_ctc = self.config['solver']['decode_ctc_weight'] >0
        if self.config['solver']['decode_ctc_weight'] >0:
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.verbose('Joint CTC decoding is enabled with weight = '+str(self.config['solver']['decode_ctc_weight']))
            self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
            self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']
            
        # Enable joint RNNLM decoding
        self.decode_lm = self.config['solver']['decode_lm_weight'] >0
        setattr(self.asr_model,'decode_lm_weight',self.config['solver']['decode_lm_weight'])
        if self.decode_lm:
            assert os.path.exists(self.config['solver']['decode_lm_path']), 'Please specify RNNLM path.'
            self.asr_model.load_lm(**self.config['solver'])
            self.verbose('Joint RNNLM decoding is enabled with weight = '+str(self.config['solver']['decode_lm_weight']))
            self.verbose('Loading RNNLM from '+self.config['solver']['decode_lm_path'])
            self.decode_file += '_lm{:}'.format(self.config['solver']['decode_lm_weight'])
        
        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model.clear_att()
        self.asr_model = self.asr_model.to(self.device)
        self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        self.valid()
        self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
        ## self.test_set = [(x,y) for (x,y) in self.test_set][::10]
        _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
        
        self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
        
        self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
                                                    str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
        
    def write_hyp(self,hyps,y):
        '''Record decoding results'''
        gt = self.mapper.translate(y,return_string=True)
        # Best
        with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
            best_hyp = self.mapper.translate(hyps[0].outIndex,return_string=True)
            f.write(gt+'\t'+best_hyp+'\n')
        # N best
        with open(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'),'a') as f:
            for hyp in hyps:
                best_hyp = self.mapper.translate(hyp.outIndex,return_string=True)
                f.write(gt+'\t'+best_hyp+'\n')
        

    def beam_decode(self,x,y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device = self.device,dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model
        
        self.write_hyp(hyps,y)
        del hyps
        
        return 1

    
    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        val_cer = 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        ctc_results = []
        with torch.no_grad():
            for cur_b,(x,y) in enumerate(self.dev_set):
                self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.dev_set)),')']))

                # Prepare data
                if len(x.shape)==4: x = x.squeeze(0)
                if len(y.shape)==3: y = y.squeeze(0)
                x = x.to(device = self.device,dtype=torch.float32)
                y = y.to(device = self.device,dtype=torch.long)
                state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # Forward
                ctc_pred, state_len, att_pred, att_maps, _ = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)
                ctc_pred = torch.argmax(ctc_pred,dim=-1).cpu() if ctc_pred is not None else None
                ctc_results.append(ctc_pred)

                # Result
                label = y[:,1:ans_len+1].contiguous()
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
                val_len += int(x.shape[0])
        
        
        # Dump model score to ensure model is corrected
        self.verbose('Validation Error Rate of Current model : {:.4f}      '.format(val_cer/val_len)) 
        self.verbose('See {} for validation results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
        with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
            for hyp,gt in zip(all_pred,all_true):
                f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
        
        # Also dump CTC result if available
        if ctc_results[0] is not None:
            ctc_results = [i for ins in ctc_results for i in ins]
            ctc_text = []
            for pred in ctc_results:
                p = [i for i in pred.tolist() if i != 0]
                p = [k for k, g in itertools.groupby(p)]
                ctc_text.append(self.mapper.translate(p,return_string=True))
            self.verbose('Also, see {} for CTC validation results.'.format(os.path.join(self.ckpdir,'dev_ctc_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_ctc_decode.txt'),'w') as f:
                for hyp,gt in zip(ctc_text,all_true):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')


class RNNLM_Trainer(Solver):
    ''' Trainer for RNN-LM only'''
    def __init__(self, config, paras):
        super(RNNLM_Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_dev_ppx = 1000

        # training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.apex = config['solver']['apex']
        print("XYZ")

    def load_data(self):
        ''' Load training / dev set'''
        self.verbose('Loading text data from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=True,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=True,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Setup RNNLM'''
        self.verbose('Init RNNLM model.')
        self.rnnlm = RNN_LM(out_dim=self.mapper.get_dim(),**self.config['rnn_lm']['model_para'])
        self.rnnlm = self.rnnlm.to(self.device)

        

        # optimizer
        if self.apex and self.config['rnn_lm']['optimizer']['type']=='Adam':
            import apex
            self.rnnlm_opt = apex.optimizers.FusedAdam(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'])
        else:
            self.rnnlm_opt = getattr(torch.optim,self.config['rnn_lm']['optimizer']['type'])
            self.rnnlm_opt = self.rnnlm_opt(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'],eps=1e-8)

        if self.paras.load:
            #raise NotImplementedError
            checkpoint = torch.load(os.path.join(self.ckpdir,'rnnlm'))
            self.rnnlm.load_state_dict(checkpoint['model_state_dict'])
            self.rnnlm_opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.loss = checkpoint['loss']
            self.rnnlm.train()
            print("model loaded")

    def exec(self):
        ''' Training RNN-LM'''
        self.verbose('RNN-LM Training set total '+str(len(self.train_set))+' batches.')

        while self.step < self.max_step:
            for y in self.train_set:
                self.progress('Training step - '+str(self.step))
                # Load data
                if len(y.shape)==3: y = y.squeeze(0)
                y = y.to(device = self.device,dtype=torch.long)
                ans_len = torch.sum(y!=0,dim=-1)

                self.rnnlm_opt.zero_grad()
                _, prob = self.rnnlm(y[:,:-1],ans_len)
                self.loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
                self.loss.backward()
                #self.loss = loss
                self.rnnlm_opt.step()

                # logger
                ppx = torch.exp(self.loss.cpu()).item()
                self.log.add_scalars('perplexity',{'train':ppx},self.step)

                # Next step
                self.step += 1
                if self.step % self.valid_step ==0:
                    self.valid()
                if self.step > self.max_step:
                    break

    def valid(self):
        self.rnnlm.eval()

        print_loss = 0.0
        dev_size = 0 

        for cur_b,y in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.dev_set)),')']))
            if len(y.shape)==3: y = y.squeeze(0)
            y = y.to(device = self.device,dtype=torch.long)
            ans_len = torch.sum(y!=0,dim=-1)
            _, prob = self.rnnlm(y[:,:-1],ans_len)
            loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
            print_loss += loss.clone().detach() * y.shape[0]
            dev_size += y.shape[0]

        print_loss /= dev_size
        dev_ppx = torch.exp(print_loss).cpu().item()
        self.log.add_scalars('perplexity',{'dev':dev_ppx},self.step)
        
        # Store model with the best perplexity
        if dev_ppx < self.best_dev_ppx:
            self.best_dev_ppx  = dev_ppx
            self.verbose('Best val ppx      : {:.4f}       @ step {}'.format(self.best_dev_ppx,self.step))
            #torch.save(self.rnnlm,os.path.join(self.ckpdir,'rnnlm'))
            torch.save({
            'step': self.step,
            'model_state_dict': self.rnnlm.state_dict(),
            'optimizer_state_dict': self.rnnlm_opt.state_dict(),
            'loss': self.loss,
            }, os.path.join(self.ckpdir,'rnnlm'))

        self.rnnlm.train()