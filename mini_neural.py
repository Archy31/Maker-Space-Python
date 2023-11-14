from numpy import exp, array, random, dot
#тренировочные данные
training_set_inputs = array([
    [0, 0, 1], 
    [1, 1, 1], 
    [1, 0, 1], 
    [0, 1, 1]])
#результат данных
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
#создание весов 
synaptic_weights = 2 * random.random((3, 1)) - 1
neuron=dot(training_set_inputs, synaptic_weights)
#тренировка
for iteration in range(10000):
    output = 1 / (1 + exp(-(neuron)))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
#проверка на возможность предсказания
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))