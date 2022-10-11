import torch.nn.functional as F
import torch.optim as optim
import os, joblib
from trainer.il.il_wrapper import *
TIME_DEBUG = False
from NuriUtils.debug_utils import log_time
from torch.autograd import Variable
import torch.nn as nn


class ImgGoalTrainer(nn.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.agent = agent
        self.env_wrapper = eval(cfg.IL.WRAPPER)(None)
        self.feature_dim = cfg.features.visual_feature_dim
        self.torch_device = 'cuda:' + str(cfg.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'
        self.optim = optim.Adam(
            list(filter(lambda p: p.requires_grad,self.agent.parameters())),
            lr=cfg.IL.lr
        )
        self.config = cfg
        self.env_setup_done = False
        self.graph_dir = cfg['ARGS']['prebuild_path']

    def save(self,file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)

    def forward(self, batch, train=True):
        train_info, aux_info, vis_info = batch
        for obs in train_info:
            if obs not in ["scene", "data_path"]:
                train_info[obs] = train_info[obs].to(dtype=torch.float).to(device=self.torch_device)
        for i in range(len(train_info['object'])):
            train_info['object'][i, :, :, 0] = i
        if train_info.get('target_object', None) != None:
            for i in range(len(train_info['target_object'])):
                train_info['target_object'][i, :, :, 0] = i
        if train_info.get('target_loc_object', None) != None:
            for i in range(len(train_info['target_loc_object'])):
                train_info['target_loc_object'][i, :, :, 0] = i
        aux_info = {'have_been': aux_info['have_been'].to(self.torch_device), 'progress': aux_info['progress'].to(self.torch_device),
                    'have_seen': aux_info['have_seen'].to(self.torch_device), 'is_obj_target': aux_info['is_obj_target'].to(self.torch_device), 'is_img_target': aux_info['is_img_target'].to(self.torch_device)}
        self.B = train_info['action'].shape[0]
        lengths = (train_info['action'] > -10).sum(dim=1)

        T = lengths.max().item()
        hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B, self.agent.net._hidden_size).to(self.torch_device)
        actions = torch.zeros([self.B]).to(self.torch_device)
        if TIME_DEBUG: s = log_time()
        split = "train" if train else "val"
        results = {}

        with torch.no_grad(): #batch x T x []
            gt_action = Variable(train_info['action'][:, :T])
            valid_indices = gt_action.long() != -100
            valid_obj_indices = (train_info['object_mask'][:, :T].long() != 0) * (valid_indices[...,None])
            gt_have_been = Variable(aux_info['have_been'][:, :T])
            gt_progress = Variable(aux_info['progress'][:, :T])
            gt_have_seen = Variable(aux_info['have_seen'][:, :T])
            gt_is_img_target = Variable(aux_info['is_img_target'][:, :T])

        graphs = []
        for data_path in train_info['data_path']:
            file_name = os.path.join(self.graph_dir, split, data_path.split('/')[-1])
            graphs.append(joblib.load(file_name)['graph'])
        actions_logits_all = []
        have_been_pred = []
        progress_pred = []
        have_seen_pred = []
        is_img_target_pred = []
        for t in range(T):
            masks = lengths > t
            if t == 0: masks[:] = False
            train_info_t = {}
            len_data = train_info['panoramic_rgb'].shape[1]
            for obs in train_info:
                try:
                    if train_info[obs].shape[1] == len_data:
                        train_info_t[obs] = train_info[obs][:, t]
                except:
                    pass
            obs_t = self.env_wrapper.step([train_info_t, torch.ones(self.B).to(self.torch_device)*t, (~masks).cpu().detach().numpy()], graphs)
            if TIME_DEBUG : s, get_step_t = log_time(s, 'env step', return_time=True)
            if -100 in actions:
                b = torch.where(actions==-100)
                for b_ in b:
                    b_ = b_.to(self.torch_device)
                    actions[b_] = 0
            obs_t.update({k: v[:, t] for k,v in aux_info.items()})
            (
                values,
                pred_act,
                actions_log_probs,
                hidden_states,
                actions_logits,
                preds,
                _
            ) = self.agent.act(
                obs_t,
                hidden_states,
                actions.view(self.B,1),
                masks.unsqueeze(1)
            )
            actions_logits_all.append(actions_logits)
            progress_pred.append(preds[0].squeeze(-1))
            is_img_target_pred.append(preds[1].squeeze(-1))
        actions_logits_all = torch.stack(actions_logits_all, 1)
        progress_pred = torch.stack(progress_pred, 1)
        is_img_target_pred = torch.stack(is_img_target_pred, 1)

        action_loss = F.cross_entropy(actions_logits_all.reshape(-1,actions_logits_all.shape[-1]), gt_action.reshape(-1).long())
        progress_loss = F.mse_loss(torch.sigmoid(progress_pred)[valid_indices].reshape(-1), gt_progress[valid_indices].reshape(-1).float())
        goal_loss = F.mse_loss(torch.sigmoid(is_img_target_pred)[valid_indices].reshape(-1), gt_is_img_target[valid_indices].reshape(-1).float())
        total_loss = action_loss + progress_loss + goal_loss

        if train:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        loss_dict = {}
        loss_dict['act_loss'] = action_loss.item()
        loss_dict['have_been'] = aux_loss1.item()
        loss_dict['progress'] = aux_loss2.item()
        loss_dict['have_seen'] = aux_loss3.item()
        loss_dict['is_target'] = aux_loss4.item()
        return results, loss_dict
