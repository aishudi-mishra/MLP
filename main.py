from test.unit_network_testbench import unit_network

inputs = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]
output = [
    0,
    1,
    1
]
learning_rate = 0.0001
epochs = 50000
unit_network(inputs, output, learning_rate, epochs)