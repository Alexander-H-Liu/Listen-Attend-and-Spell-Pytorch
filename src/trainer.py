import os
import torch
import math
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from src.asr import Seq2Seq
from src.dataset import LoadDataset
from src.postprocess import Mapper,cal_acc,cal_cer,draw_att

VAL_STEP = 30    # Additional Inference Timesteps to run during validation (to calculate CER)
DEBUG_STEP = 250 # steps for debugging info.
GRAD_CLIP = 3

class Trainer():
    def __init__(self,config,paras):
        
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

        # Logger Settings
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.ckpdir = paras.ckpdir
        if not os.path.exists(paras.ckpdir):os.makedirs(paras.ckpdir)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['trainer']['valid_step']
        self.valid_metric = config['trainer']['valid_metric']
        self.best_val_acc = 0.0
        self.best_val_ed = 2.0

        # training details
        self.step = 0
        self.max_step = config['trainer']['total_steps']
        self.tf_start = config['trainer']['tf_start']
        self.tf_end = config['trainer']['tf_end']
        self.apex = config['trainer']['apex']

        self.mapper = Mapper(config['trainer']['data_path'])




    def load_data(self):
        ''' Load training / dev set'''
        print('[INFO] Loading data from ',self.config['trainer']['data_path'])
        setattr(self,'train_set',LoadDataset('train',use_gpu=self.paras.gpu,**self.config['trainer']))
        setattr(self,'dev_set',LoadDataset('dev',use_gpu=self.paras.gpu,**self.config['trainer']))
        setattr(self,'test_set',LoadDataset('test',use_gpu=self.paras.gpu,**self.config['trainer']))
        for self.sample_x,_ in getattr(self,'train_set'):break
        if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

    def set_model(self):
        ''' Setup ASR / CLM (if enabled)'''
        print('[INFO] Init ASR model, note that Error Rate computed at validation step is Attention decoder with greedy decoding.')
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model'])
        self.asr_model.to(self.device)
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']
        if self.paras.load:
            self.asr_model.load_state_dict(torch.load(self.paras.load))

        # optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)


    def train(self):
        ''' Training End-to-end ASR system'''
        print('[INFO] Training set total',len(self.train_set),'batches.')

        while self.step< self.max_step:
            for x,y in self.train_set:
                print('Training step -',self.step,'                              ',end='\r')
                
                # Perform teacher forcing rate decaying
                tf_rate = self.tf_start - self.step*(self.tf_start-self.tf_end)/self.max_step
                
                # Hack bucket, record state length for each uttr, get longest label seq for decode step
                assert len(x.shape)==4,'Bucketing should cause acoustic feature to have shape 1xBxTxD'
                assert len(y.shape)==3,'Bucketing should cause label have to shape 1xBxT'
                x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                state_len = np.sum(np.sum(x.cpu().data.numpy(),axis=-1)!=0,axis=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # ASR forwarding 
                self.asr_opt.zero_grad()
                ctc_pred, state_len, att_pred, _ =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)

                # Calculate loss function
                loss_log = {}
                label = y[:,1:ans_len+1].contiguous()
                ctc_loss = 0
                att_loss = 0

                if self.ctc_weight<1:
                    b,t,c = att_pred.shape
                    att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))#.view(b,t)
                    att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1).to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                    att_loss = torch.mean(att_loss) # Mean by batch
                    loss_log['train_att'] = att_loss
                if self.ctc_weight>0:
                    target_len = torch.sum(y!=0,dim=-1)
                    ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                    loss_log['train_ctc'] = ctc_loss
                
                asr_loss = (1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss
                loss_log['train_full'] = asr_loss

                
                # Backprop
                asr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    print('Error - grad norm is NaN @ step',self.step)
                else:
                    self.asr_opt.step()
                
                # Logger
                self.write_log('loss',loss_log)
                if self.ctc_weight<1:
                    self.write_log('acc',{'train':cal_acc(att_pred,label)})
                if self.step % DEBUG_STEP ==0:
                    self.write_log(self.valid_metric,{'train':cal_cer(att_pred,label,metric=self.valid_metric,mapper=self.mapper)})

                # Validation
                if self.step%self.valid_step == 0:
                    self.asr_opt.zero_grad()
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
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        self.asr_model.eval()
        
        val_loss, val_ctc, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0
        val_len = 0    
        all_pred,all_true = [],[]

        for cur_b,(x,y) in enumerate(self.dev_set):
            print('Valid step -',self.step,'(',cur_b,'/',len(self.dev_set),')',end='\r')

            # Prepare data
            if len(x.shape)==4: x = x.squeeze(0)
            if len(y.shape)==3: y = y.squeeze(0)
            x = x.to(device = self.device,dtype=torch.float32)
            y = y.to(device = self.device,dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Loss
            label = y[:,1:ans_len+1].contiguous()
            if self.ctc_weight<1:
                seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
                seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1).to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                seq_loss = torch.mean(seq_loss) # Mean by batch
                val_att += seq_loss.detach()*int(x.shape[0])
                t1,t2 = cal_cer(att_pred,label,metric=self.valid_metric,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_acc += cal_acc(att_pred,label)*int(x.shape[0])
                val_cer += cal_cer(att_pred,label,metric=self.valid_metric,mapper=self.mapper)*int(x.shape[0])
            
            if self.ctc_weight>0:
                target_len = torch.sum(y!=0,dim=-1)
                ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                val_ctc += ctc_loss.detach()*int(x.shape[0])

            val_len += int(x.shape[0])
        
        # Logger
        val_loss = (1-self.ctc_weight)*val_att + self.ctc_weight*val_ctc
        loss_log = {}
        for k,v in zip(['dev_full','dev_ctc','dev_att'],[val_loss, val_ctc, val_att]):
            if v > 0.0: loss_log[k] = v/val_len
        self.write_log('loss',loss_log)

        if self.ctc_weight<1:
            val_hyp,val_txt = cal_cer(att_pred,label,metric=self.valid_metric,mapper=self.mapper,get_sentence=True)
            val_attmap = draw_att(att_maps,att_pred)

            self.write_log(self.valid_metric,{'dev':val_cer/val_len})
            self.write_log('acc',{'dev':val_acc/val_len})
            for idx,attmap in enumerate(val_attmap):
                self.write_log('att_'+str(idx),attmap)
                self.write_log('hyp_'+str(idx),val_hyp[idx])
                self.write_log('txt_'+str(idx),val_txt[idx])        

            # checkpoint by val acc.
            if val_acc/val_len > self.best_val_acc:
                self.best_val_acc = val_acc/val_len
                print('Best val acc      : {:.4f}       @ step {}'.format(self.best_val_acc,self.step))
                torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir,self.exp_name+'_asr_acc.model'))

            # checkpoint by val er.
            if val_cer/val_len  < self.best_val_ed:
                self.best_val_ed = val_cer/val_len
                print('Best val er       : {:.4f}       @ step {}'.format(self.best_val_ed,self.step))
                torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir,self.exp_name+'_asr_er.model'))
                # Save hyps.
                with open(os.path.join(self.ckpdir,self.exp_name+'_hyp.txt'),'w') as f:
                    for t1,t2 in zip(all_pred,all_true):
                        f.write(t1+','+t2+'\n')

        self.asr_model.train()

    def inference(self):
        '''Perform inference step with beam search attention decoding.'''
        self.asr_model.eval()

        test_cer = 0.0
        all_pred, all_true = [], []
        print('Start testing with beam decode...')

        for cur_b, (x,y) in enumerate(tqdm(self.test_set)):
            # Prepare data
            if len(x.shape)==4: x = x.squeeze(0)
            if len(y.shape)==3: y = y.squeeze(0)
            x = x.to(device = self.device,dtype=torch.float32)
            y = y.to(device = self.device,dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            with torch.no_grad():
                output = self.asr_model.beam_decode(x, ans_len+VAL_STEP, state_len, 20)

            label = y[:,1:ans_len+1].contiguous()
            att_pred = output[0].outIndex
            att_pred = torch.LongTensor(att_pred).unsqueeze(0).to(self.device)
            t1, t2 = cal_cer(att_pred, label, metric=self.valid_metric, mapper=self.mapper, get_sentence=True, argmax=False)
            all_pred += t1
            all_true += t2
            test_cer += cal_cer(att_pred, label, metric=self.valid_metric, mapper=self.mapper, argmax=False)*int(x.shape[0])
            # for o in output:
                # print("score: {:.4f} -- {}".format(o.avgScore(), o.outIndex))

        test_len = len(self.test_set.dataset)
        print('Test cer:', (test_cer/test_len))
