from fsl_dataloader import Dataset
from protonet import ProtoNet, FSL_ProtoNet
from maml import MAML

config={
    "n_way":5,
    "num_support":2,
    "num_query":1,
    "learning_rate":0.001,
    "batch_size":16,
    "epochs":15000,
    "val_iter":1
}

ds = Dataset(config)
train_data, val_data, test_data = ds.get_data(1) 
train_ds = ds.get_dataloader(train_data)
val_ds = ds.get_dataloader(val_data)
test_ds = ds.get_dataloader(test_data)

#Train Protonet
# model=ProtoNet()
# fsl = FSL_ProtoNet(model, config)
# fsl.train(train_ds, val_ds)
# fsl.test(test_ds)

#Train MAML
maml = MAML(5,1)
maml.train(train_ds, val_ds,True, True)