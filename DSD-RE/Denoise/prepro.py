from config_parser import create_config
from reader.reader import init_dataset
from dataset import PreTrainDataset

configFilePath = "config/PreTraining.config"
config = create_config(configFilePath)
PreTrainDataset(config, "train").prepro_data(config)