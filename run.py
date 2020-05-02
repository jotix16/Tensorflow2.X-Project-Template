#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sam May  2 15:25:34 2020

@author: mikelzhobro
"""
import sys
sys.dont_write_bytecode = True

from config import base_config

from compilation_options.optimizer import get_optimizer
from compilation_options.loss import get_loss_function
from compilation_options.metrics import get_metrics_lst
from compilation_options.callback import get_callbacks
from models.model import get_model

from trainer import Trainer
from data_loader.data import get_datasets

if __name__ == "__main__":
    
    config = base_config()
    config.METRICS_LST = get_metrics_lst()
    config.OPTIMIZER = get_optimizer()
    config.LOSS_FUNC = get_loss_function()
    config.CALLBACK_LST = get_callbacks(config)
    
    config.display()

    model       = get_model(config)
    datasets    = get_datasets(config)
    trainer     = Trainer(datasets, model, config)

    trainer._compile()
    trainer.train()
