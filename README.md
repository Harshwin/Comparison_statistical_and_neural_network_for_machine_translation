# Comparison-of-Statistical-and-neural-machine-learning

This repository contains a comparison of statistical approaches and neural network based approaches performance for the task of Machine translation. This was our teams paper analysis for ECE657. Josh Reid, Harshwin Venugopal, Sruthy Paul, Ruoxuan {fjs2reid; hvenugopal; s25paul; ruoxuan.xu}@uwaterloo.ca

nmt

Creating a Neural Machine Translator for English-French


Installation

Download the French-English corpus from http://www.manythings.org/anki/
and put it into the corpra folder. Run the script prep_data.py to clean and split
the data into training and testing sets. Then run trainer.py to train the
neural machine translator and finally run tester.py with your own text within the
variable 'phrase' to translate it. Currently does French to English, but loading
other datasets from that link should work as well as long as the prep_data.py file
is changed to get that file instead.
