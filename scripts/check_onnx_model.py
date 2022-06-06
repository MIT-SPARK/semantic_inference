import sys
import onnx #onnx required downgrading numpy from 1.19.5 to 1.19.4

#load model
if len(sys.argv) != 2:
	print("Error! Must provide path to .onnx file as argument.")
	exit()

print("Loading model:", sys.argv[1])
model = onnx.load(sys.argv[1])

#check if model valid
try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')

#show input names
input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))
print('Input names: ', net_feed_input)

#show output names
output =[node.name for node in model.graph.output]
print('Output names: ', output)