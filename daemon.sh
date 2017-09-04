#!/bin/bash

PREV=0
for I in 0.2 0.4 0.6 0.8 1; do
	nohup src/interpolationbyparts.py $PREV $I &
	PREV=$I
done
