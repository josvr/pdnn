#!/bin/bash

rm *.log

./AdaptiveLearningRateTest.sh
./ConstantLearningRateTest.sh
./DropOutTest.sh
./MaxOutTest.sh
./RectifierDropOutTest.sh
./SdATest.sh



