import os
import tempfile

from twisted.internet import threads
from twisted.internet.defer import Deferred, inlineCallbacks


class Evolve(object):
    
    def __init__(self, network, mutation_fn, fitness_fn, input_perc_fn, input_data, expected_results, population=100, top_percent=0.1, name=None):
        self.generation = 0
        self.population = population
        self.set_top_percent(top_percent)
        self.mutation_fn = mutation_fn
        self.fitness_fn = fitness_fn
        self.input_perc_fn = input_perc_fn
        self.replace_data(input_data, expected_results)
        
        self.name = name or type(self).__name__
        
        self.populate_generation(selected_networks=[network])
    
    def set_top_percent(self, top_percent):
        self.top_percent = top_percent
        self.top_fit = int(self.population * top_percent)
        if self.top_fit < 1:
            self.top_fit = 1
    
    def replace_data(self, input_data, expected_results):
        self.desired_outputs = []
        for expected in expected_results:
            new_desired = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            new_desired[expected] = 1
            self.desired_outputs.append(new_desired)
        self.expected_results = expected_results
        self.input_data = input_data
        self.input_data_length = len(input_data)
    
    def replace_fitness_fn(self, fitness_fn):
        self.fitness_fn = fitness_fn
    
    def get_fittest_networks(self, autosave=True):
        networks = []
        for index in self.fittest_indices:
            networks.append(self.networks[index])
        if autosave:
            index, f_score, cost = self.fitness_results[0]
            filename = "%s_%.0f.json" % (self.name, f_score * 100)
            #filename = os.path.join(tempfile.gettempdir(), filename)
            filename = os.path.join("Autosaves", filename)
            try:
                self.networks[index].save(filename)
            except FileNotFoundError as e:
                print("[WARN] autosave failed: %s" % e)
        return networks
    
    def populate_generation(self, selected_networks=[], mutate=True):
        self.networks = []
        self.generation += 1
        
        selected_networks_length = len(selected_networks)
        
        self.networks.extend(selected_networks)
        
        for index in range(self.population - selected_networks_length):
            network = selected_networks[index % selected_networks_length].copy()
            if mutate:
                self.mutation_fn(network)
            self.networks.append(network)
    
    @inlineCallbacks
    def run_generation(self):
        self.fittest_indices = []
        
        fitness_results = yield self.__thread_networks_run(self.networks)
        #fitness_results.sort(reverse=True, key=Evolve.key_sort_results)
        fitness_results.sort(key=Evolve.key_sort_results)
        
        _, self.fitness_best, _ = fitness_results[0]
        self.fitness_avg = 0
        for index, f_score, cost in fitness_results:
            self.fitness_avg += f_score
        self.fitness_avg /= self.population
        
        for t_index in range(self.top_fit):
            index, f_score, cost = fitness_results[t_index]
            self.fittest_indices.append(index)
        
        self.fitness_results = fitness_results
        return self.fitness_results
    
    def __thread_networks_run(self, network):
        d = Deferred()
        fitness_results = []
        input_data_normal = self.input_perc_fn(self.input_data)
        for index in range(self.population):
            network = self.networks[index]
            resp = threads.deferToThread(network.run, input_data_normal)
            resp.addCallback(self.__thread_networks_output_handler(fitness_results, index, d))
        return d
    
    def __thread_networks_output_handler(self, fitness_results, network_index, d_trigger):
        def handler(outputs):
            cost = self.networks[network_index].cost(self.desired_outputs)
            f_score = self.calculate_fitness(outputs)
            #fitness_results.append((network_index, f_score))
            fitness_results.append((network_index, f_score, cost))
            # Catch last f_score and trigger the yielded deferred (d_trigger) with final fitness result list
            if len(fitness_results) == self.population:
                d_trigger.callback(fitness_results)
        return handler
    
    def calculate_fitness(self, outputs):
        total_score = 0.0
        n_outputs = len(outputs)
        
        for index in range(n_outputs):
            exp_output = self.expected_results[index]
            output = outputs[index]
            total_score += self.fitness_fn(output.tolist(), exp_output)
        
        return total_score / n_outputs
    
    def key_sort_results(item):
        index, score, cost = item
        return cost
