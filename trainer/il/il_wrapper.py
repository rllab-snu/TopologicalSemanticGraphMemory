import torch
try:
     from gym.wrappers.monitor import Wrapper
except:
     from gym.wrappers.record_video import RecordVideo as Wrapper
TIME_DEBUG = False

class ILWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,env):
        super().__init__(env)
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_batch_graph(self, graphs, t=0):
        graph_t = []
        for graph, t_ in zip(graphs, t):
            t_ = int(t_)
            if len(graph) > t_:
                graph_t.append(graph[t_])
            else:
                graph_t.append(graph[-1])
        memory_dict = {}
        for key, val in graph_t[0].items():
            if isinstance(val, int):
                memory_dict[key] = torch.tensor([graph[key] for graph in graph_t]).to(self.torch_device)
            else:
                if "A" in key: #for affinity matrix
                    max_num_node_a = max([graph[key].shape[0] for graph in graph_t])
                    max_num_node_b = max([graph[key].shape[1] for graph in graph_t])
                    graph_placeholder = torch.zeros([len(graphs), max_num_node_a, max_num_node_b])
                    for graph_i, graph in enumerate(graph_t):
                        num_node_a = graph[key].shape[0]
                        num_node_b = graph[key].shape[1]
                        graph_placeholder[graph_i, :num_node_a,:num_node_b] = torch.from_numpy(graph[key])[:num_node_a,:num_node_b][None]
                    memory_dict[key] = graph_placeholder.to(self.torch_device)
                elif 'img_memory_idx' in key:
                    pass
                else:
                    max_num_node = max([graph[key].shape[0] for graph in graph_t])
                    graph_placeholder = torch.zeros([len(graphs), max_num_node, *val.shape[1:]])
                    for graph_i, graph in enumerate(graph_t):
                        num_node = len(graph[key])
                        graph_placeholder[graph_i, :num_node] = torch.from_numpy(graph[key])[:num_node][None]
                    memory_dict[key] = graph_placeholder.to(self.torch_device)
        return memory_dict

    def step(self, batch, graphs=None):
        with torch.no_grad():
            obs_batch, t, done_list = batch
            obs_batch['step'] = t
            graphs = self.get_batch_graph(graphs, t)
            obs_batch.update(graphs)
        return obs_batch
