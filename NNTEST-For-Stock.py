# cd OneDrive\Documents\PythonScripts
# python NNTEST-For-Stock.py

import random, numpy, copy, math
from os import listdir
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Settings - does not use GPU but CPU - so its slow - so don't put to many big numbers for the Neural Networks and their Neurons
START_HIDDEN_NEURONS = 20
HALF_HIDDEN_NEURONS = round(START_HIDDEN_NEURONS / 2)
MAX_HIDDEN_NEURONS = 22
MAX_LAYERS = 5
MAX_NN = 20
CONTROL_NN = 1
MUTATIONS = 5 # Mutations each generation
EPOCHS = 10
LR = 0.01
MODE = 0 # Mode 0: Guess Exact Price / Mode 1: Guess Whether Price will go up or down (at time of TO_PREDICT)
SAMPLE_SIZE = 1000 # Days in a sequence for one TRAIN/TEST TIME 
TO_PREDICT = 1 # Amount of Days out to predict
TRAIN_TIME = 5 
TEST_TIME = 1
EVO_TIME = 100

#Don't Touch
INPUT_AMOUNT = 1
OUTPUT_AMOUNT = 1

sc = MinMaxScaler(feature_range=(0,1))
sc3 = MinMaxScaler(feature_range=(0,1))

NN_ids = 0

#########################################################################################################

class Neuron():
  def __init__(self, inputs, owner, id):
    self.owner = owner
    self.activated_num = 0
    self.pre_activated_num = 0 # If not part of main Neural Network; In order to make sure Neuron can store data
    self.weighted_sum = 0
    self.weights = []
    
    # Used to figure out if Neuron is a part of main Neural Network
    self.layer = 0
    self.rlayer = 0
    self.prio = 0
    self.derived_weights = []
    self.local_layered = []
    self.local_rlayered = []
    self.layer_done = False
    self.rlayer_done = False
    

    for input in range(len(inputs) + 1):
      self.weights.append(2 * random.random() - 1)

    self.inputs = inputs
    self.outputs = []
    self.id = id
    self.owner.active_ids.append(id)

  def set_outputs(self):
    for input in self.inputs:
      self.owner.neurons[input].outputs.append(self.id)

  def set_layer(self): # Used to check if Neuron is a part of main Neural Network if the Neurons inputs were also checked (Forward)
    if not(self.id in self.owner.cur_layered):
      self.layer = len(self.owner.neurons)
  
    self.owner.cur_layered.append(self.id)
    for input in self.inputs:
      if not(input in self.local_layered):
        input_neuron = self.owner.neurons[input]
        self.local_layered.append(input_neuron.id)
        if input in self.owner.cur_layered and input_neuron.layer_done == False:
          input_neuron.set_layer()

        if not(input in self.owner.cur_layered):
          input_neuron.set_layer()

        if input_neuron.layer < self.layer:
          self.layer = input_neuron.layer

    if self.layer_done == False:
      self.layer += 1
      self.prio += self.layer
      self.layer_done = True

  def set_rlayer(self): # Used to check if Neuron is a part of main Neural Network if the Neurons outputs were also checked (Backward)
    if not(self.id in self.owner.cur_rlayered):
      self.rlayer = len(self.owner.neurons)
      
    self.owner.cur_rlayered.append(self.id)
    for output in self.outputs:
      if not(output in self.local_rlayered):
        output_neuron = self.owner.neurons[output]
        self.local_rlayered.append(output_neuron.id)
        if output in self.owner.cur_rlayered and output_neuron.rlayer_done == False:
          output_neuron.set_rlayer()
          
        if not(output in self.owner.cur_rlayered):
          output_neuron.set_rlayer()

        if output_neuron.rlayer < self.rlayer:
          self.rlayer = output_neuron.rlayer

    if self.rlayer_done == False:
      self.rlayer += 1
      self.prio += self.rlayer
      self.rlayer_done = True

  def set_derived_weights(self):
    self.derived_weights = []
    for weight in self.weights:
      self.derived_weights.append([])
      
    self.weights

  def activate(self): # Activates Neuron by first checking if its inputs are also activated
    self.owner.cur_activated.append(self.id)
    self.weighted_sum = 0
    index = 0
    if self.owner.priority != self.prio:
      for output in self.outputs:
        if not(output in self.owner.cur_activated):
          output_neuron = self.owner.neurons[output]
          output_neuron.pre_activated_num = output_neuron.activated_num
          output_neuron.activate()

    for input in self.inputs:
      input_neuron = self.owner.neurons[input]
      if not(input in self.owner.cur_activated) and input_neuron.prio == self.owner.priority:
        input_neuron.activate()

      self.weighted_sum += input_neuron.activated_num * self.weights[index]
      index += 1

      # If Neuron is not apart of the main Neural Network, it stores previous data to be used for the next day in the sequence
      if not(input in self.owner.cur_activated):
        input_neuron.pre_activated_num = input_neuron.activated_num
        input_neuron.activate()

    self.weighted_sum += self.weights[index]
    self.activated_num = 2 / (1 + numpy.exp(-self.weighted_sum)) - 1

  def derive(self): # Derives weights of Neuron by first checking if its outputs are also derived
    self.owner.cur_derived.append(self.id)
    self.derived_chain = 1
    if self.id == 0:
      self.derived_chain *= -self.owner.derived_cost

    for output in self.outputs:
      output_neuron = self.owner.neurons[output]
      if not(output in self.owner.cur_derived):
        output_neuron.derive()

      i = 0
      for input in output_neuron.inputs:
        if input == self.id:
          break
        i += 1

      self.derived_chain += (output_neuron.weights[i] * output_neuron.sigmoid_derived() * 
        output_neuron.derived_chain)

    if self.id != 0:
      self.derived_chain -= 1

    for input in range(len(self.inputs)):
      input_neuron = self.owner.neurons[self.inputs[input]]
      if self.owner.priority == input_neuron.prio:
        self.derived_weights[input].append(input_neuron.activated_num * self.sigmoid_derived() * 
          self.derived_chain)

      else:
        self.derived_weights[input].append(input_neuron.pre_activated_num * self.sigmoid_derived() * 
          self.derived_chain) # uses pre_activated_num which stores previous data

    self.derived_weights[len(self.derived_weights) - 1].append(self.sigmoid_derived() * self.derived_chain)

  def sigmoid_derived(self):
    derived = 2 * numpy.exp(self.weighted_sum) / pow(1 + numpy.exp(self.weighted_sum), 2)
    return derived

  def change_weights(self):
    for weight in range(len(self.weights)):
      epoch_weights = self.derived_weights[weight][:EPOCHS]
      average = sum(epoch_weights) / len(epoch_weights)
      for epoch in range(EPOCHS):
        self.derived_weights[weight].pop(0)

      self.weights[weight] += average * LR
      if self.weights[weight] > 1:
        self.weights[weight] = 1
      elif self.weights[weight] < -1:
        self.weights[weight] = -1

#########################################################################################################

class Neural_Network():
  def __init__(self, id):
    self.id = id
    self.active_ids = []
    self.account = Account(self)
    self.predicted = []
    self.real = []
    self.money_line = []
    self.cur_activated = []
    self.cur_derived = []
    self.neurons = []
    self.fitness_num = 0
    self.mutations = []
    
    # Builds starting Neural Network
    for input in range(INPUT_AMOUNT):
      self.neurons.append(Neuron([], self, len(self.neurons) + OUTPUT_AMOUNT))

    neuron_amount = random.randrange(HALF_HIDDEN_NEURONS, START_HIDDEN_NEURONS)

    temp_layers = [INPUT_AMOUNT]
    for i in range(round(neuron_amount / (START_HIDDEN_NEURONS / MAX_LAYERS)), 0, -1):
      temp_layers_amount = round(neuron_amount / i)
      if temp_layers_amount == 0:
        temp_layers_amount += 1

      temp_layers.append(temp_layers_amount)
      neuron_amount -= temp_layers_amount

    temp_layers.append(OUTPUT_AMOUNT)

    count = 0
    for layer in range(1, len(temp_layers)):
      temp_inputs = []
      for neuron in range(count, temp_layers[layer - 1] + count):
        if layer == 1:
          temp_inputs.append(self.neurons[neuron].id)
        else:
          temp_inputs.append(self.neurons[neuron].id)

        count += 1
      if not(layer == len(temp_layers) - 1):
        for neuron in range(temp_layers[layer]):
          self.neurons.append(Neuron(copy.copy(temp_inputs), self, len(self.neurons) + OUTPUT_AMOUNT))
      else:
        for neuron in range(temp_layers[layer]):
          self.neurons.insert(0, Neuron(copy.copy(temp_inputs), self, neuron + temp_layers[layer] - 1))

    for neuron in self.neurons:
      neuron.set_outputs()

  def set_prios(self): # Sets prio for each Neuron by calling to corresponding function in the neurons
    print(self.id)
    for neuron in self.neurons:
      neuron.prio = 0
      neuron.local_layered = []
      neuron.local_rlayered = []
      neuron.layer_done = False
      neuron.rlayer_done = False
      
    self.cur_layered = [OUTPUT_AMOUNT]
    self.neurons[OUTPUT_AMOUNT].layer = 0
    self.neurons[OUTPUT_AMOUNT].layer_done = True
    
    self.cur_rlayered = [0]
    self.neurons[0].rlayer = 0
    self.neurons[0].rlayer_done = True
    
    self.priority = len(self.neurons)
    for neu in self.active_ids:
      neuron = self.neurons[neu]
      if not(neuron.id in self.cur_layered):
        neuron.set_layer() # function
      if not(neuron.id in self.cur_rlayered):
        neuron.set_rlayer() # function

      if self.priority > neuron.prio:
        self.priority = neuron.prio

      print(str(neuron.layer) + " L " + str(neuron.rlayer) + " R " + str(neuron.prio) + ": " + 
        str(neuron.id) + ", " + str(self.priority) + " P")

  def set_derived_weights(self):
    for neuron in self.active_ids:
      self.neurons[neuron].set_derived_weights()

  def train(self, ins_outs):
    scale = sc.fit_transform(ins_outs)
    ins_outs = sc.transform(ins_outs)
    self.epoch = 0
    self.real = []
    self.predicted = []
    self.account.money = 0
    self.account.stock = 0
    self.money_line = []
    for in_out in ins_outs: # Trains on each input and correct output pair
      self.epoch += 1
      self.cur_activated = [OUTPUT_AMOUNT]
      self.neurons[OUTPUT_AMOUNT].activated_num = in_out[0]
      for neuron in self.active_ids:
        if not(neuron in self.cur_activated):
          self.neurons[neuron].activate()


      predictedP = self.neurons[0].activated_num
      self.real.append(in_out[1])
      self.account.process_money(in_out, predictedP)

      self.derived_cost = 2 * (self.neurons[0].activated_num - in_out[1])
      self.cur_derived = []
      for neuron in self.active_ids:
        if not(neuron in self.cur_derived):
          self.neurons[neuron].derive()

      if self.epoch % EPOCHS == 0 and self.epoch >= TO_PREDICT:
        for neuron in self.active_ids:
          self.neurons[neuron].change_weights()

  def test(self, ins_outs):
    self.ins_outs = []
    for in_out in ins_outs:
      self.ins_outs.append(in_out[0])
      
    scale = sc.fit_transform(ins_outs)
    ins_outs = sc.transform(ins_outs)
    self.predicted = []
    self.account.money = 0
    self.account.stock = 0
    self.money_line = []
    self.fitness_num = 0
    for in_out in ins_outs: # Tested on each input and correct output pair to get a fitness score
      self.cur_activated = [OUTPUT_AMOUNT]
      self.neurons[OUTPUT_AMOUNT].activated_num = in_out[0]
      for neu in self.active_ids:
        neuron = self.neurons[neu]
        if not(neuron.id in self.cur_activated):
          neuron.activate()

      predictedP = self.neurons[0].activated_num
      if MODE == 0:
        self.predicted.append([in_out[0], predictedP])
      else:
        self.predicted.append(predictedP)

      self.account.process_money(in_out, predictedP)

    # Fitness score
    self.fitness_num = (sum(self.money_line) / sum(self.account.cur_price) * ((self.account.total / 
      self.account.start) / (self.account.cur_price[len(self.account.cur_price) - 1] / 
      self.account.cur_price[0])))

  def show_graph(self):
    if MODE == 0:
      predicted_in = sc.inverse_transform(self.predicted)
      predicted_price = []
      for pre in predicted_in:
        predicted_price.append(pre[1])
        
      for price in range(len(predicted_price)):
        predicted_price[price] *= 10
          
      plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
    else:
      for price in range(len(self.predicted)):
          self.predicted[price] *= 100
      plt.plot(self.predicted, color = 'green', label = 'Predicted Stock Growth Direction')
    
    for price in range(len(self.account.cur_price)):
      self.ins_outs[price] *=  10
    
    plt.plot(self.money_line, color = 'orange', label = 'Money')
    plt.plot(self.ins_outs, color = 'purple', label = 'Real Price')
    plt.title(str(self.id) + ' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


  def get_fitness(self):
    return self.fitness_num


  # A Bunch of functions to modify the Neural Network
  def add_weight(self, _from, _to):
    neuron_to = self.neurons[_to]
    if not(_from in neuron_to.inputs):
      neuron_to.inputs.append(_from)
      neuron_to.weights.append(2 * random.random() - 1)
      self.neurons[_from].outputs.append(_to)

  def remove_weight(self, _from, _to):
    neuron_to = self.neurons[_to]
    if _from in neuron_to.inputs:
      neuron_from = self.neurons[_from]
      index = neuron_to.inputs.index(_from)
      neuron_to.inputs.pop(index)
      neuron_to.weights.pop(index)
      index = neuron_from.outputs.index(_to)
      neuron_from.outputs.pop(index)
      
      if neuron_to.inputs == [] and neuron_to.outputs == [] and neuron_to.id in self.active_ids:
        self.active_ids.pop(self.active_ids.index(neuron_to.id))
      if neuron_from.inputs == [] and neuron_from.outputs == [] and neuron_from.id in self.active_ids:
        self.active_ids.pop(self.active_ids.index(neuron_from.id))

  def add_neuron(self):
    for neu in range(round(len(self.neurons) / (START_HIDDEN_NEURONS / MAX_LAYERS)) - 1):
      inputs = [neu + 2]

      neuron = Neuron(inputs, self, len(self.neurons))
      neuron.set_outputs()
      self.neurons.append(neuron)

      self.add_weight(neuron.id, neu + 2)

  def add_neuron1(self):
    inputs = []
    for input in range(1, random.randrange(round(len(self.neurons)/5))):
      id = self.active_ids[random.randrange(1, len(self.active_ids))]
      in_ = id in inputs
      while in_ == True:
        id = self.active_ids[random.randrange(1, len(self.active_ids))]
        in_ = id in inputs
      inputs.append(id)

    neuron = Neuron(inputs, self, len(self.neurons))
    neuron.set_outputs()
    self.neurons.append(neuron)

    for output in range(1, random.randrange(round(len(self.neurons)/5))):
      id = self.active_ids[random.randrange(1, len(self.active_ids))]
      in_ = id in neuron.outputs
      while in_ == True:
        id = self.active_ids[random.randrange(1, len(self.active_ids))]
        in_ = id in neuron.outputs
      self.add_weight(neuron.id, id)

  def remove_neuron(self, index):
    neuron = self.neurons[index]
    for input in neuron.inputs:
      self.remove_weight(input, neuron.id)

    for output in neuron.outputs:
      self.remove_weight(neuron.id, output)

#########################################################################################################

class Account(): # Gives Neural Network money to test if it know when to buy or sell
  def __init__(self, owner):
    self.owner = owner
    self.cur_price = []
    self.money = 0
    self.start = 0
    self.stock = 0
    self.total = 0
    self.max = 0
    self.end = 0

  def process_money(self, in_out, predict):
      re_in_out = in_out.reshape(1, -1)
      self.cur_price.append(float(sc.inverse_transform(re_in_out)[0][0]))

      self.last_index = self.cur_price[len(self.cur_price) - 1]

      if self.money == 0 and self.cur_price != 0:
        self.money = self.last_index * 10
        self.start = self.money

      if MODE == 0:
        if (self.money > self.last_index and in_out[0] < predict):
          self.buy()
        elif self.stock > 0 and in_out[0] > predict:
          self.sell()
      else:
        if (self.money > self.last_index and predict > 0):
          self.buy()
        elif self.stock > 0 and predict < 0:
          self.sell()

      self.total = self.money + self.last_index * self.stock
      if self.max < self.total:
        self.max = self.total

      self.owner.money_line.append(self.total)
      #print(str(self.total) + ": " + str(self.owner.id))

  def buy(self):
    while self.money > self.last_index:
      self.money -= self.last_index
      self.stock += 1

  def sell(self):
    while self.stock > 0:
      self.money += self.last_index
      self.stock -= 1

#########################################################################################################

if __name__ == "__main__":
   # Path to my data if you want to test this out you would want to change this
  path = "C:\\Users\\grink\\OneDrive\\Documents\\PythonScripts\\DailyStockOHLC"
  list_ticker = listdir(path)

  amount = len(list_ticker)
  ticker = 0

##################################


  def get_in_out_pair():
    lines = []
    f = 0
    complete_path = ""
    while f == 0: # Random ticker that works
      try:
        ticker = random.randrange(amount)
        complete_path = path + "\\" + list_ticker[ticker]
        f = open(complete_path, "r")
        lines = f.readlines()
        f.close()
        if not(len(lines) > SAMPLE_SIZE + TO_PREDICT): 
          f = 0

      except:
        pass

    print(list_ticker[ticker])
    numbers = []

    print(len(lines))
    if len(lines) > SAMPLE_SIZE + TO_PREDICT:
      position = 1
      for j in lines:
        for d in j.split():
          mod = position % 5
          if mod == 4:
            numbers.append(float(d))

          position += 1

      in_out_pair = []
      if MODE == 0:
        for price in range(SAMPLE_SIZE):
          in_out_pair.append([numbers[price], numbers[price + TO_PREDICT]])
      else:
        for price in range(SAMPLE_SIZE):
          cur_price = numbers[price]
          to_predict = numbers[price + TO_PREDICT]
          if to_predict > cur_price:
            in_out_pair.append([cur_price, 1])
          elif to_predict < cur_price:
            in_out_pair.append([cur_price, -1])
          else:
            in_out_pair.append([cur_price, 0])

      return in_out_pair # Input and correct output pair

##################################


  def train_all():
    for neu_net in neural_networks:
      neu_net.set_prios()
      
    for neu_net in control_networks:
      neu_net.set_prios()
    
    for train in range(TRAIN_TIME):
      in_out_pair = get_in_out_pair()
      for neu_net in neural_networks:
        neu_net.set_derived_weights()
        neu_net.account.cur_price = []
        neu_net.train(in_out_pair)
      
      for neu_net in control_networks:
        neu_net.set_derived_weights()
        neu_net.account.cur_price = []
        neu_net.train(in_out_pair)

##################################

  def test_all(): # Test all to find which one is the best
    normalize_fit = []
    for neu_net in neural_networks:
      normalize_fit.append(0)
      
    for neu_net in control_networks:
      normalize_fit.append(0)

    for test in range(TEST_TIME):
      fits = []
      in_out_pair = get_in_out_pair()
      for neu_net in neural_networks:
        neu_net.account.cur_price = []
        neu_net.test(in_out_pair)
        fits.append([neu_net.fitness_num])
        
      for neu_net in control_networks:
        neu_net.account.cur_price = []
        neu_net.test(in_out_pair)
        fits.append([neu_net.fitness_num])

      scale1 = sc3.fit_transform(fits)
      fits = sc3.transform(fits)

      for neu_net in range(len(neural_networks) + len(control_networks)):
        normalize_fit[neu_net] += fits[neu_net][0]

      for net_num in range(len(neural_networks) + len(control_networks)):
        if net_num < len(neural_networks):
          neu_net = neural_networks[net_num]
        else:
          neu_net = control_networks[net_num - len(neural_networks)]
          
        print(str(normalize_fit[net_num]) + " - " + str((neu_net.account.total / neu_net.account.start) / 
          (neu_net.account.cur_price[len(neu_net.account.cur_price) - 1] / neu_net.account.cur_price[0])) 
            + ": " + str(neu_net.id))

    for neu_net in range(len(neural_networks) + len(control_networks)):
      if neu_net < len(neural_networks):
        neural_networks[neu_net].fitness_num = normalize_fit[neu_net]
      else:
        control_networks[neu_net - len(neural_networks)].fitness_num = normalize_fit[neu_net]

##################################


  def next_generation(): # Evolves best Neural Networks
    neural_networks.sort(key=Neural_Network.get_fitness)
    neural_networks[len(neural_networks) - 1].show_graph()

    half = len(neural_networks) / 2
    nets_to = 0
    SCALE = 10
    for index in range(math.floor(half)):
      rand = -random.random()
      if rand < 1 / (1 + numpy.exp(index - math.floor(half)) / (MAX_NN / SCALE)) - 1:
        neural_networks.pop(index)
        nets_to += 1

    while nets_to > 0:
      rand_index = random.randrange(math.floor(half)) + round(half) - nets_to
      rand_per = random.random()
      if rand_per > 2 / (1 + numpy.exp(rand_index) / (MAX_NN / SCALE)) - 1:
        duplicate(neural_networks[rand_index])
        nets_to -= 1

##################################


  def duplicate(neural_network): # Duplicates best Neural Networks with mutations
    new_network = copy.deepcopy(neural_network)
    new_network.id = globals()['NN_ids'] + 1
    globals()['NN_ids'] += 1
    active_ids = new_network.active_ids
    for mutation in range(1):
      mutate = random.randrange(4)
      match mutate:
        case 0:
          new_network.add_weight(active_ids[random.randrange(len(active_ids))], 
            active_ids[random.randrange(len(active_ids))])
        case 1:
          new_network.remove_weight(active_ids[random.randrange(len(active_ids))], 
            active_ids[random.randrange(len(active_ids))])
        case 2:
          if len(new_network.active_ids) != MAX_HIDDEN_NEURONS:
            new_network.add_neuron()
          else:
            new_network.remove_neuron(active_ids[random.randrange(len(active_ids))])
            mutate = 3
        case 3:
          new_network.remove_neuron(active_ids[random.randrange(len(active_ids))])
          
      new_network.mutations.append(mutate)

    neural_networks.append(new_network)
    #for neuron in new_network.neurons:
     # print(str(neuron.inputs) + " " + str(neuron.id))

##################################

  neural_networks = []
  for neu_net in range(MAX_NN):
    neural_networks.append(Neural_Network(len(neural_networks)))
    NN_ids += 1
  
  control_networks = []
  for neu_net in range(CONTROL_NN):
    control_networks.append(Neural_Network(len(neural_networks) + len(control_networks)))

  for evol in range(EVO_TIME):
    train_all()
    test_all()
    next_generation()













