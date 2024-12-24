import pickle
import torch as th
from collections import defaultdict

class TableModel():
    def __init__(self, args):
        self.args = args
        self.d = defaultdict(dict)
        self.args.epoch_learn_world_model = 1
        
    def update(self, batch):
        states  = batch['states'].type(th.int64)  
        actions = batch['actions'].type(th.int64)  
        inputs  = th.cat([states, actions], dim=1).cpu().numpy()
        rewards = batch['rewards']  
        bs = states.shape[0]
        for b in range(bs):
            dict_key = tuple(inputs[b])
            for i, obj in enumerate(self.args.learning_obj):
                if obj not in self.d[dict_key]:
                    self.d[dict_key][obj] = rewards[b][i].item()

    def predict_rewards(self, inputs, forward_obj):
         
        bs = inputs.shape[0]
        inputs = inputs.cpu().numpy()
        out = []
        for b in range(bs):
            dict_key = tuple(inputs[b])
            if dict_key in self.d:
                b_out = []
                for obj in forward_obj:
                    if obj in self.d[dict_key]:
                        b_out.append(self.d[dict_key][obj])
                    else:
                        b_out.append(0)
                out.append(th.tensor(b_out).unsqueeze(0))
            else:
                out.append(th.zeros(1, len(forward_obj)))
                
        out = th.cat(out, dim=0).to(self.args.device)
        return out

    def save_models(self, path):
        with open(f'{path}/table_world_model.pkl', 'wb') as file:
            pickle.dump(self.d, file)

    def load_models(self, path):
        with open(f'{path}/table_world_model.pkl', 'rb') as file:
            self.d = pickle.load(file)
