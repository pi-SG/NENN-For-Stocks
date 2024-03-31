package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
)

// Settings - does not use GPU but CPU - so its slow - so dont put to many big numbers for the Neural Networks and their Neurons - The Neural Network seems faster on Go, though

var START_HIDDEN_NEURONS int = 200
var HALF_HIDDEN_NEURONS int = int(math.Round(float64(START_HIDDEN_NEURONS / 2)))
var MAX_HIDDEN_NEURONS int = 100
var MAX_LAYERS int = 5
var EPOCHS int = 10
var LR float64 = .002
var MODE int = 0          // Mode 0: Guess Exact Price / Mode 1: Guess Whether Price will go up or down (at time of TO_PREDICT)
var SAMPLE_SIZE int = 100 // Days in a sequence for one TRAIN/TEST TIME
var TO_PREDICT int = 1    // Amount of Days out to predict
var TRAIN_TIME int = 25
var TEST_TIME int = 5
var INPUT_AMOUNT int = 5

// Keep Constant
var OUTPUT_AMOUNT int = 1

//sc = MinMaxScaler(feature_range=(0,1))
//sc3 = MinMaxScaler(feature_range=(0,1))

var NN_ids int = 0

//////////////////////////////////////////////////////////////////////////////////////////////////////////

type Neuron struct {
	owner                                      *Neural_Network
	id                                         int
	inputs, outputs                            []int
	activated_num, weighted_sum, derived_chain float64
	weights                                    []float64
	derived_weights                            [][]float64
}

func Make_Neuron(o *Neural_Network, id int, in []int) *Neuron {
	o.active_ids = append(o.active_ids, id)

	var w []float64
	for input := 0; input < len(in)+1; input++ {
		w = append(w, rand_range_float64(-1, 1))
	}

	var f float64
	var ssf [][]float64
	var i []int
	if in == nil {
		return &Neuron{o, id, i, i, f, f, f, w, ssf}
	}
	return &Neuron{o, id, in, i, f, f, f, w, ssf}
}

type Neuron_Interface interface {
	set_outputs()
	activate()
	set_derived_weights()
}

func (neuron *Neuron) set_outputs() {
	for input := 0; input < len(neuron.inputs); input++ {
		neuron.owner.neurons[neuron.inputs[input]].outputs = append(neuron.owner.neurons[neuron.inputs[input]].outputs, neuron.id)
	}
}

func (neuron *Neuron) set_derived_weights() {
	neuron.derived_weights = [][]float64{}
	for wei := 0; wei < len(neuron.weights); wei++ {
		neuron.derived_weights = append(neuron.derived_weights, []float64{})
	}
}

func (neuron *Neuron) activate() { // Activates Neuron by first checking if its inputs are also activated
	neuron.owner.cur_activated = append(neuron.owner.cur_activated, neuron.id)
	neuron.weighted_sum = 0
	/*if neuron.owner.priority != neuron.prio{
		for output := range neuron.outputs {
			if !slices.Contains(neuron.owner.cur_activated, output) {
				output_neuron := neuron.owner.neurons[output]
				//output_neuron.pre_activated_num = output_neuron.activated_num
				output_neuron.activate()
			}
		}
	}*/

	for input := 0; input < len(neuron.inputs); input++ {
		input_neuron := neuron.owner.neurons[neuron.inputs[input]]
		if !slices.Contains(neuron.owner.cur_activated, input_neuron.id) { //&& input_neuron.prio == neuron.owner.priority:
			input_neuron.activate()
		}
		neuron.weighted_sum += input_neuron.activated_num * neuron.weights[input]
	}
	neuron.weighted_sum += neuron.weights[len(neuron.weights)-1]
	neuron.activated_num = 2/(1+math.Exp(-neuron.weighted_sum)) - 1
}

func (neuron *Neuron) derive() { // Derives weights of Neuron by first checking if its outputs are also derived
	neuron.owner.cur_derived = append(neuron.owner.cur_derived, neuron.id)
	neuron.derived_chain = 1
	if neuron.id == 0 {
		neuron.derived_chain *= -neuron.owner.derived_cost
	}

	for output := 0; output < len(neuron.outputs); output++ {
		output_neuron := neuron.owner.neurons[neuron.outputs[output]]
		if !slices.Contains(neuron.owner.cur_derived, output_neuron.id) {
			output_neuron.derive()
		}

		i := 0
		for input := 0; input < len(output_neuron.inputs); input++ {
			if output_neuron.inputs[input] == neuron.id {
				break
			}
			i += 1
		}
		neuron.derived_chain += (output_neuron.weights[i] * output_neuron.sigmoid_derived() * output_neuron.derived_chain)
	}

	if neuron.id != 0 {
		neuron.derived_chain -= 1
	}

	for input := 0; input < len(neuron.inputs); input++ {
		input_neuron := neuron.owner.neurons[neuron.inputs[input]]
		//if neuron.owner.priority == input_neuron.prio {
		neuron.derived_weights[input] = append(neuron.derived_weights[input], input_neuron.activated_num*neuron.sigmoid_derived()*neuron.derived_chain)
		/*} else {
			neuron.derived_weights[input].append(input_neuron.pre_activated_num * neuron.sigmoid_derived() *
				neuron.derived_chain) // uses pre_activated_num which stores previous data
		}*/
	}
	neuron.derived_weights[len(neuron.derived_weights)-1] = append(neuron.derived_weights[len(neuron.derived_weights)-1], neuron.sigmoid_derived()*neuron.derived_chain)
}

func (neuron *Neuron) sigmoid_derived() float64 {
	derived := 2 * math.Exp(neuron.weighted_sum) / math.Pow(1+math.Exp(neuron.weighted_sum), 2)
	return derived
}

func (neuron *Neuron) change_weights() {
	for weight := 0; weight < len(neuron.weights); weight++ {
		epoch_weights := neuron.derived_weights[weight][:EPOCHS]
		average := sum(epoch_weights) / float64(len(epoch_weights))
		for epoch := 0; epoch < EPOCHS; epoch++ {
			neuron.derived_weights[weight] = neuron.derived_weights[weight][1:]
		}

		neuron.weights[weight] += average * LR
		if neuron.weights[weight] > 1 {
			neuron.weights[weight] = 1
		} else if neuron.weights[weight] < -1 {
			neuron.weights[weight] = -1
		}
	}
}

type Neural_Network struct {
	neurons                                []*Neuron
	epoch                                  int
	cur_activated, cur_derived, active_ids []int
	derived_cost                           float64
	average                                []float64
}

type Neural_Network_Interface interface {
	start_network()
	set_derived_weights()
	train([]float64)
	test([]float64)
}

func (NN *Neural_Network) start_network() {
	var neurons []*Neuron
	for input := 0; input < INPUT_AMOUNT; input++ {
		neurons = append(neurons, Make_Neuron(NN, len(neurons)+OUTPUT_AMOUNT, nil))
	}

	neuron_amount := math.Round(rand_range_float64(HALF_HIDDEN_NEURONS, START_HIDDEN_NEURONS))

	temp_layers := []int{INPUT_AMOUNT}
	for i := int(math.Round(neuron_amount / float64(START_HIDDEN_NEURONS/MAX_LAYERS))); i > 0; i-- {
		temp_layers_amount := math.Round(neuron_amount / float64(i))
		if temp_layers_amount == 0 {
			temp_layers_amount += 1
		}

		temp_layers = append(temp_layers, int(temp_layers_amount))
		neuron_amount -= temp_layers_amount
	}

	temp_layers = append(temp_layers, OUTPUT_AMOUNT)
	count := 0
	for layer := 1; layer < len(temp_layers); layer++ {
		var temp_inputs []int
		add := 0
		for neu := count; neu < temp_layers[layer-1]+count; neu++ {
			temp_inputs = append(temp_inputs, neurons[neu].id)
			add++
		}
		count += add
		if layer != len(temp_layers)-1 {
			for neu := 0; neu < temp_layers[layer]; neu++ {
				neurons = append(neurons, Make_Neuron(NN, len(neurons)+OUTPUT_AMOUNT, temp_inputs))
			}
		} else {
			for neu := 0; neu < temp_layers[layer]; neu++ {
				neurons = slices.Insert(neurons, neu, Make_Neuron(NN, neu+temp_layers[layer]-1, temp_inputs))
			}
		}
	}

	NN.neurons = neurons
	for neuron := range NN.neurons {
		NN.neurons[neuron].set_outputs()
	}
}

func (NN Neural_Network) set_derived_weights() {
	for neu := 0; neu < len(NN.active_ids); neu++ {
		NN.neurons[NN.active_ids[neu]].set_derived_weights()
	}
}

func (NN *Neural_Network) train(ins_outs_un []float64) {
	var ins_outs []float64 = ins_outs_un
	var scale []float64
	if MODE == 0 {
		ins_outs, scale = normalize(ins_outs, 0, 1)
		scale[0] += 1
	}
	//NN.account.money = 0
	//NN.account.stock = 0
	//NN.money_line = []
	var predictedP []float64
	var actual []float64
	for in_out := 0; in_out < len(ins_outs); in_out += OUTPUT_AMOUNT + INPUT_AMOUNT { // Trains on each input and correct output pair
		NN.epoch += 1
		NN.cur_activated = []int{}
		for input := OUTPUT_AMOUNT; input < OUTPUT_AMOUNT+INPUT_AMOUNT; input++ {
			NN.cur_activated = append(NN.cur_activated, NN.active_ids[input])
			NN.neurons[NN.active_ids[input]].activated_num = ins_outs[in_out+input-OUTPUT_AMOUNT]
		}
		for neu := 0; neu < len(NN.active_ids); neu++ {
			if !slices.Contains(NN.cur_activated, NN.active_ids[neu]) {
				NN.neurons[NN.active_ids[neu]].activate()
			}
		}
		fmt.Println("Predicted: " + fmt.Sprintf("%f", NN.neurons[0].activated_num) + ", Real: " + fmt.Sprintf("%f", ins_outs[in_out+1]) + ", Percent: " + fmt.Sprintf("%f", NN.neurons[0].activated_num/ins_outs[in_out+1]))

		actual = append(actual, ins_outs[in_out+INPUT_AMOUNT])
		predictedP = append(predictedP, NN.neurons[0].activated_num)

		NN.derived_cost = 2 * (NN.neurons[0].activated_num - ins_outs[in_out])
		NN.cur_derived = []int{}
		for neu := 0; neu < len(NN.active_ids); neu++ {
			if !slices.Contains(NN.cur_derived, NN.active_ids[neu]) {
				NN.neurons[NN.active_ids[neu]].derive()
			}
		}

		if NN.epoch%EPOCHS == 0 && NN.epoch >= TO_PREDICT {
			for neu := 0; neu < len(NN.active_ids); neu++ {
				NN.neurons[NN.active_ids[neu]].change_weights()
			}
		}
	}
}

func (NN *Neural_Network) test(ins_outs_un []float64) {
	var ins_outs []float64
	var scale []float64
	if MODE == 0 {
		ins_outs, scale = normalize(ins_outs_un, 0, 1)
	} else {
		for in_out := 0; in_out < len(ins_outs_un); in_out += OUTPUT_AMOUNT + INPUT_AMOUNT {
			for input := 0; input < INPUT_AMOUNT; input++ {
				ins_outs = append(ins_outs, ins_outs_un[in_out+input])
			}
		}

		var ins_outs_temp []float64
		ins_outs_temp, scale = normalize(ins_outs, 0, 1)
		scale[0] += 1
		for in_out := 0; in_out < len(ins_outs_temp); in_out++ {
			ins_outs = append(ins_outs, ins_outs_temp[in_out], ins_outs_un[in_out*2+1])
		}
	}

	//NN.account.money = 0
	//NN.account.stock = 0
	//NN.money_line = []
	//NN.fitness_num = 0
	var predictedP []float64
	for in_out := 0; in_out < len(ins_outs); in_out += OUTPUT_AMOUNT + INPUT_AMOUNT { // Tested on each input and correct output pair to get a fitness score
		for input := OUTPUT_AMOUNT; input < OUTPUT_AMOUNT+INPUT_AMOUNT; input++ {
			NN.cur_activated = append(NN.cur_activated, NN.active_ids[input])
			NN.neurons[NN.active_ids[input]].activated_num = ins_outs[in_out+input-OUTPUT_AMOUNT]
		}
		for neu := 0; neu < len(NN.active_ids); neu++ {
			if !slices.Contains(NN.cur_activated, NN.active_ids[NN.active_ids[neu]]) {
				NN.neurons[NN.active_ids[neu]].activate()
			}
		}

		predictedP = append(predictedP, NN.neurons[0].activated_num)
		//NN.account.process_money(in_out, predictedP)
		fmt.Println("Predicted: " + fmt.Sprintf("%f", NN.neurons[0].activated_num) + ", Real: " + fmt.Sprintf("%f", ins_outs[in_out+1]) + ", Percent: " + fmt.Sprintf("%f", NN.neurons[0].activated_num/ins_outs[in_out+1]) + " test")
		NN.average = append(NN.average, (NN.neurons[0].activated_num / ins_outs[in_out+1]))
	}

	// Fitness score
	//NN.fitness_num = (sum(NN.money_line) / sum(NN.account.cur_price) * ((NN.account.total /NN.account.start) / (NN.account.cur_price[len(NN.account.cur_price)-1] / NN.account.cur_price[0])))
}

func main() {
	var NN_Interface Neural_Network_Interface
	var Neural_Network Neural_Network

	NN_Interface = &Neural_Network
	NN_Interface.start_network()
	NN_Interface.set_derived_weights()
	for i := 0; i < TRAIN_TIME; i++ {
		NN_Interface.train(get_in_out_pair())
	}
	for i := 0; i < TEST_TIME; i++ {
		NN_Interface.test(get_in_out_pair())
	}
	fmt.Println(sum(Neural_Network.average) / float64(len(Neural_Network.average)))
}

// Path to my data if you want to test this out you would want to change this
var path string = "C:\\Users\\grink\\OneDrive\\Documents\\Git-Repositories\\NNTEST-For-Stocks\\DailyStockOHLC"
var list_ticker, err = os.ReadDir(path)

var amount int = len(list_ticker)

func get_in_out_pair() []float64 {
	var lines []string
	var f []byte = nil
	checked := false
	ticker := 0
	complete_path := ""
	for checked == false { // Find random ticker that works
		ticker = int(rand_range_float64(0, amount))
		complete_path = path + "\\" + list_ticker[ticker].Name()
		f, err = os.ReadFile(complete_path)
		lines = strings.Fields(string(f))
		fmt.Println(len(lines))
		if len(lines)/5 > SAMPLE_SIZE+TO_PREDICT {
			checked = true
		}
	}

	var numbers []float64
	position := 1
	for j := 0; j < len(lines); j++ {
		mod := position % 5
		if mod == 4 {
			float := 0.0
			float, err = strconv.ParseFloat(strings.TrimSpace(lines[j]), 64)
			numbers = append(numbers, float)
		}
		position++
	}

	var in_out_pair []float64
	randpoint := int(rand_range_float64(INPUT_AMOUNT, len(numbers)-SAMPLE_SIZE+TO_PREDICT-1))
	if MODE == 0 {
		for cur_price := randpoint; cur_price < SAMPLE_SIZE+randpoint; cur_price++ {
			for price := cur_price - INPUT_AMOUNT + 1; price <= cur_price; price++ {
				in_out_pair = append(in_out_pair, numbers[price])
			}
			in_out_pair = append(in_out_pair, numbers[cur_price+TO_PREDICT])
		}
	} else {
		for cur_price := randpoint; cur_price < SAMPLE_SIZE+randpoint; cur_price++ {
			for price := cur_price - INPUT_AMOUNT + 1; price <= cur_price; price++ {
				in_out_pair = append(in_out_pair, numbers[price])
			}

			to_predict := numbers[cur_price+TO_PREDICT]
			if to_predict > numbers[cur_price] {
				in_out_pair = append(in_out_pair, 1)
			} else if to_predict < numbers[cur_price] {
				in_out_pair = append(in_out_pair, -1)
			} else {
				in_out_pair = append(in_out_pair, 0)
			}
		}
	}
	return in_out_pair // Input and correct output pair (stock price data)
}

func rand_range_float64(min int, max int) float64 {
	return rand.Float64()*float64(max-min) + float64(min)
}

func normalize(s []float64, min float64, max float64) ([]float64, []float64) {
	var smin float64 = 0
	var smax float64 = 0
	for i := 0; i < len(s); i++ {
		if i == 0 || s[i] < smin {
			smin = s[i]
		}
	}
	for i := 0; i < len(s); i++ {
		if i == 0 || s[i] > smax {
			smax = s[i]
		}
	}

	scale := []float64{smin - min - .000001, (smax - smin - min - .000001) / max}
	for i := 0; i < len(s); i++ {
		s[i] = (s[i] - scale[0]) / scale[1]
	}
	return s, scale
}

func sum(sf []float64) float64 {
	sum := 0.0
	for i := 0; i < len(sf); i++ {
		sum += sf[i]
	}
	return sum
}
