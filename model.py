import torch
from TorchCRF import CRF
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning, BridgeTowerModel,RobertaModel
import torch.nn as nn
import torch.nn.functional as F


class BridgeTowerForMABSA(nn.Module):
    def __init__(self,args):
        super(BridgeTowerForMABSA, self).__init__()

        self.pretrained_model = args.pretrained_model
        self.backbone = BridgeTowerModel.from_pretrained(self.pretrained_model)
        self.momentum_backbone = BridgeTowerModel.from_pretrained(self.pretrained_model)
        self.is_adapter = args.is_adapter
        self.hidden_dim = args.hidden_dim
        self.adapter_hidden_dim = args.adapter_hidden_dim

        self.image_adapter = nn.Sequential(nn.Linear(self.hidden_dim,self.adapter_hidden_dim),nn.GELU(),nn.Linear(self.adapter_hidden_dim,self.hidden_dim))
        self.momentum_image_adapter = nn.Sequential(nn.Linear(self.hidden_dim,self.adapter_hidden_dim),nn.GELU(),nn.Linear(self.adapter_hidden_dim,self.hidden_dim))

        self.text_adapter = nn.Sequential(nn.Linear(self.hidden_dim,self.adapter_hidden_dim),nn.GELU(),nn.Linear(self.adapter_hidden_dim,self.hidden_dim))
        self.momentum_text_adapter = nn.Sequential(nn.Linear(self.hidden_dim,self.adapter_hidden_dim),nn.GELU(),nn.Linear(self.adapter_hidden_dim,self.hidden_dim))

        if self.is_adapter == True:
            self.remove_gradient()
            self.model_pairs = [
                [self.text_adapter,self.momentum_text_adapter],
                [self.image_adapter,self.momentum_image_adapter]
            ]
        else:
            self.model_pairs = [
                [self.backbone, self.momentum_backbone],
            ]


        self.queue_size = args.queue_size

        self.copy_params()
        self.temp = nn.Parameter(torch.ones([]) * args.temp)
        self.register_buffer("image_queue", torch.tensor([]))
        self.register_buffer("text_queue", torch.tensor([]))
        self.register_buffer("attention_mask_queue", torch.tensor([]))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = args.momentum
        self.num_labels = args.num_labels


        self.logits = nn.Linear(self.hidden_dim,self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.two_stage = False
        self.k = 1

        self.inter_image = nn.Linear(self.k, 1)

        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=self.hidden_dim, kdim=self.hidden_dim, vdim=self.hidden_dim, num_heads=1, batch_first=True)


        self.LayerNormalization = nn.LayerNorm(self.hidden_dim)
        self.gate_dense = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.attention_mode = args.attention_mode
        self.patch_len = args.patch_len
        self.is_attention = args.is_attention


    def remove_gradient(self):
        for name, param in self.named_parameters():
            if "backbone" in name:
                param.requires_grad = False


    def gate_attention(self,text_hidden_state,image_hidden_state,attention_mask):
        attention_mask = attention_mask.eq(1)
        image_att, _ = self.mha_layer(text_hidden_state,image_hidden_state,image_hidden_state)
        merge = torch.cat([text_hidden_state, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        multi_hidden_states = (1 - gate) * text_hidden_state + gate * image_att
        return multi_hidden_states


    def cross_attention(self,text_hidden_state,image_hidden_state,attention_mask):
        attention_mask = attention_mask.eq(1)
        multi_hidden_state,multi_hidden_attn = self.mha_layer(text_hidden_state,image_hidden_state,image_hidden_state)

        # multi_hidden_state = multi_hidden_state + text_hidden_state #residual connection
        multi_hidden_state = self.LayerNormalization(multi_hidden_state) # layernorm
        return multi_hidden_state


    def concat_attention(self,text_hidden_state,image_hidden_state,attention_mask):
        batch_size = text_hidden_state.shape[0]
        multi_hidden_state = torch.cat([text_hidden_state,image_hidden_state],dim=1)
        multi_attention_mask = torch.cat([attention_mask, torch.ones(size=(batch_size, self.patch_len), device=self.device)],
                                     dim=1)
        multi_attention_mask = multi_attention_mask.eq(1)
        multi_hidden_state,_ = self.mha_layer(multi_hidden_state,multi_hidden_state,multi_hidden_state,key_padding_mask=multi_attention_mask)

        return multi_hidden_state


    def chosen_max_contributed_patch(self, multi_hidden_state, text_hidden_state, image_hidden_state):
        value = multi_hidden_state.topk(self.k)[0]
        index = multi_hidden_state.topk(self.k)[1]

        if self.k == 1:
            picked_image_state = torch.gather(image_hidden_state, dim=1, index=index.repeat(1, 1, self.hidden_dim))
            weighted_image_state = picked_image_state * value
            multi_hidden_state = text_hidden_state + weighted_image_state
        else:
            picked_image_state = torch.gather(image_hidden_state.unsqueeze(2).repeat(1, 1, self.k, 1), dim=1,
                                              index=index.unsqueeze(3).repeat(1, 1, 1, self.hidden_dim))
            weighted_image_state = picked_image_state * value.unsqueeze(3)
            weighted_image_state = weighted_image_state.transpose(2, 3)
            weighted_image_state = self.inter_image(weighted_image_state)
            weighted_image_state = weighted_image_state.squeeze(3)
            multi_hidden_state = text_hidden_state + weighted_image_state
        return multi_hidden_state


    def forward(self,input_ids,attention_mask,pixel_values,pixel_mask,labels=None):
        output = self.backbone(input_ids=input_ids,attention_mask=attention_mask,pixel_values=pixel_values,pixel_mask=pixel_mask)
        text_hidden_state = output.text_features
        image_hidden_state = output.image_features
        batch_size = text_hidden_state.shape[0]

        if self.is_adapter:
            text_hidden_state = self.text_adapter(text_hidden_state)
            image_hidden_state = self.image_adapter(image_hidden_state)

        if self.two_stage:
            multi_hidden_state = torch.matmul(text_hidden_state, image_hidden_state.transpose(1, 2))
            multi_hidden_state = multi_hidden_state / (image_hidden_state.size()[-1] ** 0.5)
            multi_hidden_state = nn.Softmax(dim=-1)(multi_hidden_state)

            if self.is_attention:
                if self.attention_mode == "weighted_based_addition":
                    multi_hidden_state = self.chosen_max_contributed_patch(multi_hidden_state, text_hidden_state,
                                                                       image_hidden_state)
                elif self.attention_mode == "cross_attention":
                    multi_hidden_state = self.cross_attention(text_hidden_state,image_hidden_state,attention_mask)

                elif self.attention_mode == "gate_attention":
                    multi_hidden_state = self.gate_attention(text_hidden_state,image_hidden_state,attention_mask)

                elif self.attention_mode == "concat_attention":
                    multi_hidden_state = self.concat_attention(text_hidden_state,image_hidden_state,attention_mask)

                else:
                    raise AttributeError("do not designate attention mode")
            else:
                multi_hidden_state = text_hidden_state

            logits = self.logits(multi_hidden_state)

            if self.attention_mode == "concat_attention":
                crf_mask = torch.cat([attention_mask, torch.zeros(size=(batch_size, self.patch_len), device=self.device)],
                                     dim=1)
                crf_mask = crf_mask.eq(1)
            else:
                crf_mask = attention_mask.eq(1)

            if labels is not None:
                crf_loss = -self.crf(logits, labels, crf_mask)
                return crf_loss
            else:
                return logits

        else:
            cl_loss = self.MOCO(input_ids,attention_mask,pixel_values,pixel_mask,text_hidden_state,image_hidden_state)
            return cl_loss


    def MOCO(self, input_ids,attention_mask,pixel_values,pixel_mask,text_hidden_state,image_hidden_state):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat = image_hidden_state
        text_feat = text_hidden_state

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            output = self.momentum_backbone(input_ids=input_ids,attention_mask=attention_mask,pixel_values=pixel_values,pixel_mask=pixel_mask)

            image_feat_m = output.image_features
            if self.is_adapter:
                image_feat_m = self.image_adapter(image_feat_m)

            image_feat_all = torch.cat([image_feat_m, self.image_queue.clone().detach()], dim=0)

            text_feat_m = output.text_features
            if self.is_adapter:
                text_feat_m = self.text_adapter(text_feat_m)

            text_feat_all = torch.cat([text_feat_m, self.text_queue.clone().detach()], dim=0)

        t_attention_mask = torch.cat([attention_mask, self.attention_mask_queue.clone().detach()]).to(self.device)

        i2t_loss1, i2t_loss2 = self.token_with_patch_contrastive_loss(text_feat_all, image_feat, t_attention_mask,
                                                                      dim=0)
        t2i_loss1, t2i_loss2 = self.token_with_patch_contrastive_loss(text_feat, image_feat_all, attention_mask, dim=-1)

        loss_i2t = i2t_loss1 + i2t_loss2
        loss_t2i = t2i_loss1 + t2i_loss2
        loss_itc = (loss_i2t + loss_t2i) / 2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, attention_mask)

        return loss_itc


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, attention_mask):
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)
        if self.text_queue.shape[0] >= self.queue_size:

            if ptr + batch_size > self.text_queue.shape[0]:
                t = self.text_queue.shape[0] - ptr
                self.image_queue[ptr:, :, :] = image_feat[:t, :, :]
                self.text_queue[ptr:, :, :] = text_feat[:t, :, :]
                self.attention_mask_queue[ptr:, :] = attention_mask[:t, :]

                self.image_queue[:batch_size - t, :, :] = image_feat[t:batch_size, :, :]
                self.text_queue[:batch_size - t, :, :] = text_feat[t:batch_size, :, :]
                self.attention_mask_queue[:batch_size - t, :] = attention_mask[t:batch_size, :]
                ptr = batch_size - t
            # # replace the keys at ptr (dequeue and enqueue)
            else:
                self.image_queue[ptr:ptr + batch_size, :, :] = image_feat
                self.text_queue[ptr:ptr + batch_size, :, :] = text_feat
                self.attention_mask_queue[ptr:ptr + batch_size, :] = attention_mask
                ptr = (ptr + batch_size) % self.text_queue.shape[0]  # move pointer

        else:
            self.image_queue = torch.cat([self.image_queue, image_feat])
            self.text_queue = torch.cat([self.text_queue, text_feat])
            self.attention_mask_queue = torch.cat([self.attention_mask_queue, attention_mask])

            ptr = (ptr + batch_size) % self.text_queue.shape[0]  # move pointer
        self.queue_ptr[0] = ptr


    def token_with_patch_contrastive_loss(self, text_hidden_state, image_hidden_state, attention_mask, dim):
        t2i_sim = []
        i2t_sim = []
        for i in range(len(text_hidden_state)):
            temp1 = []
            temp2 = []
            for j in range(len(image_hidden_state)):
                sim1 = self.t2i_compute_similarity(text_hidden_state[i], image_hidden_state[j], attention_mask[i])
                temp1.append(sim1.item())
                sim2 = self.i2t_compute_similarity(text_hidden_state[i], image_hidden_state[j], attention_mask[i])
                temp2.append(sim2.item())
            t2i_sim.append(temp1)
            i2t_sim.append(temp2)

        text2img_sim = torch.tensor(t2i_sim, device=self.device) / self.temp
        img2text_sim = torch.tensor(i2t_sim, device=self.device) / self.temp

        y_true = torch.zeros(text2img_sim.size()).to(self.device)
        y_true.fill_diagonal_(1)
        y_true = y_true.long()
        t2i_loss = -torch.sum(F.log_softmax(text2img_sim, dim=dim) * y_true, dim=1).mean()
        i2t_loss = -torch.sum(F.log_softmax(img2text_sim, dim=dim) * y_true, dim=1).mean()
        return t2i_loss, i2t_loss


    def t2i_compute_similarity(self, text_hidden_state, image_hidden_state, attention_mask):
        multi_hidden_state = torch.matmul(text_hidden_state, image_hidden_state.transpose(0, 1))
        text2img_max_score = multi_hidden_state.max(-1)[0].float()
        padding_length = sum(attention_mask)
        text2img_mean_score = text2img_max_score * attention_mask
        text2img_mean_score = text2img_mean_score.sum(dim=-1)
        text2img_similarity = text2img_mean_score / padding_length
        return text2img_similarity


    def i2t_compute_similarity(self, text_hidden_state, image_hidden_state, attention_mask):
        multi_hidden_state = torch.matmul(image_hidden_state, text_hidden_state.transpose(0, 1))
        img2text_max_score = multi_hidden_state * attention_mask
        img2text_mean_score = img2text_max_score.max(-1)[0].float()
        img2text_mean_score = img2text_mean_score.mean(dim=-1)
        return img2text_mean_score