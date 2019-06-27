import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
import flat
from collections import OrderedDict


def apply_weight_sharing(model, bits=8, dev = "cuda"):
    """
    Applies weight sharing to the given model
    """
    # for module in model.children():
    for name,param in model.named_parameters():
        print(name, param.data.shape, param.data.numel())

        shape = param.data.shape
        weight = param.data.cpu().numpy()

        min_ = np.min(weight)
        max_ = np.max(weight)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(weight.reshape(-1,1))
        # print("KMeans end")

        # print("kmeans.labels_", kmeans.labels_)
        # print("kmeans.cluster_centers_", kmeans.cluster_centers_)

        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        # print("new_weight", new_weight)

        # print("param.data[:5]", param.data[:5])
        param.data = torch.from_numpy(new_weight).to(dev).resize_(shape)
        # print("param.data[:5]", param.data[:5])



def flatten_model(model, bits=1, dev = "cuda"):

    l = get_flattened_module_structure(model)
    model_flat = flat.Flat(l)

    return model_flat



def get_flattened_module_structure(model, dev = "cuda"):

    l = [(name.replace(".","-"),module) for name,module in model.named_modules() if (type(module) != torch.nn.Sequential and type(module) != type(model))]
    return l


def print_module_structure(model, dev = "cuda"):

    l = [(name,module) for name,module in model.named_modules() if (type(module) != torch.nn.Sequential and type(module) != type(model))]
    print("\nmodule structure\n")
    print("len(l)", len(l))
    for n,m in l:
        print(n,type(m),m)


def check_copy_model(model):

    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())


def get_formatted_state_dict(model):

    sd = model.state_dict()
    sd_formatted = OrderedDict()
    for key in sd.keys():
        sd_formatted[key.replace(".","-",1)] = sd[key]


    return sd_formatted









