#!/bin/bash
torchserve --start --ncs --model-store . --models my_fancy_model=my_fancy_model.mar
