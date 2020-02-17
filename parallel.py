import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue

'''
Class used for represent a city.
A city have a (x,y) position and
can calculate a distance to another city.
'''
class City:
    def __init__(self, x, y):
        self.x, self.y = x, y

    '''
    Calculate a distance to another city
    '''
    def distance(self, city):

        xDis = abs(self.x - city.x) # Distance in the x axis
        yDis = abs(self.y - city.y) # Distance in the y axis

        # the euclidean distance between the cities
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(%d,%d)" % (self.x, self.y)

'''
Class used to calculate the Fitness from a route.
The fitness is calculated with the full distance's route.
'''
class Fitness:

    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    '''
    Calculate the total distance from a route.
    '''
    def routeDistance(self):

        # If the distance was not calculated
        if self.distance == 0:
            pathDistance = 0

            # Go in every city on the route
            for i in range(len(self.route)):

                fromCity = self.route[i] # Actual city
                toCity = None # Next city

                # If is the last city, the next one is the first
                if i + 1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                
                # Add the distance between the cities
                pathDistance += fromCity.distance(toCity)
            
            # Update the distance
            self.distance = pathDistance
        
        return self.distance

# cria um conjunto de cidades
def cria_cidades(qt_cidades):
    cityList = []
    for i in range(qt_cidades):
        cityList.append(
            City(
                x=int(random.random() * 200), 
                y=int(random.random() * 200)
            )
        )
    
    return cityList

# Cria uma rota de cidades
def cria_rota(cidades):
    return random.sample(cidades, len(cidades))

# inicializa uma população de rotas
def init_pop(pop_size, cidades):
    pop = []

    # Cria uma população inicial
    for i in range(pop_size):
        rota = cria_rota(cidades)
        data = {
            "rota": rota,
            "distancia": Fitness(rota).routeDistance(), 
        }

        pop.append(data)

    # Retorna a população ordenada
    pop.sort(key=lambda x: x['distancia'])

    return pop

# Cruzamento dos pais
def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    # Intervalos dos genes
    geneA = int(random.random() * len(parent1['rota']))
    geneB = int(random.random() * len(parent1['rota']))

    # Intervalo de inicio e fim
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    # esse intervalo pega caracteristicas do pai 1
    for i in range(startGene, endGene):
        childP1.append(parent1['rota'][i])
    
    # Adiciona o resto do pai 2
    childP2 = [item for item in parent2['rota'] if item not in childP1]

    # Retorna o novo filho
    rota = childP1 + childP2
    data = {
        "rota": rota,
        "distancia": -1,
    }

    return data

# Mutação do filho
def mutate(kid):

    rota = kid['rota']

    # checa todas as cidades
    for swapped in range(len(rota)):

        # troque essa cidade com uma outra cidade aleatoria
        if random.random() < 0.3:
            swapWith = int(random.random() * len(rota))

            city1 = rota[swapped]
            city2 = rota[swapWith]

            rota[swapped] = city2
            rota[swapWith] = city1
    
    # Atualiza a rota
    kid['rota'] = rota

    return kid

# cria os N_PROCESSOS para criar crianças
def process_kids(melhor, qt):

    # Operação feita por cada processo
    def single_process(qt, melhor, kids_queue):
        
        # cria qt crianças
        for i in range(qt):

            # Cria uma copia do melhor
            best = copy.deepcopy(melhor)

            # Faz mutações do melhor
            kid = mutate(best)

            # Atualiza sua distancia
            kid['distancia'] = Fitness(kid['rota']).routeDistance()
            kids_queue.put(kid)
    
    # Fila que vai manter as crianças
    kids_queue = Queue()

    # Decide a quantidade de crianças que cada processo vai gerar
    proc_size = int(qt/N_PROCS)
    proc_sizes = [proc_size for i in range(N_PROCS)]
    for i in range(qt % N_PROCS):
        proc_sizes[i] += 1
    
    # Adiciona a tarefa pra cada processo
    procs = []
    for i in range(N_PROCS):
        
        # Cria um processo 
        p = Process(
            target=single_process,
            args=(
                proc_sizes[i],
                melhor,
                kids_queue,
            )
        )

        # Adiciona na lista de processos
        procs.append(p)
        p.start()
    
    # Espera os processos finalizarem
    for p in procs:
        p.join()

    del procs

    # Pega os filhos da fila
    kids = []
    while not kids_queue.empty():
        kids.append(kids_queue.get())
    
    return kids

# Cria os filhos
def create_kids(pop):

    # cria 1/4 de filhos
    qt = int(len(pop)/4)
    melhor = pop[0]
    melhor = copy.deepcopy(melhor)
    kids = process_kids(melhor, qt)
    return kids

# Mata os piores e mantem os melhores
def kill_bad(pop, kids):
    new_pop = pop + kids
    new_pop.sort(key=lambda data:data['distancia'])
    new_pop = new_pop[:len(pop)]
    new_pop = copy.deepcopy(new_pop)

    return new_pop


def parallelGA(pop_size, qt_cidades, n_gen, info=False):
    progresso = []

    # tempo no inicio
    begin = time.time()

    # cria um grupo de cidades
    cidades = cria_cidades(qt_cidades)

    # inicializa a população
    pop = init_pop(pop_size, cidades)

    # Adiciona o progresso
    melhor = pop[0]
    progresso.append(melhor['distancia'])

    # Percorre pelo número de gerações
    for _ in tqdm(range(n_gen)):

        # Criando uma nova população
        kids = create_kids(pop) # cria filho dos melhores
        pop = kill_bad(pop, kids) # mata os piores

        # Adiciona o progresso
        progresso.append(pop[0]['distancia'])
    
    if info:
        plt.plot(progresso)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()

    # Tempo no fim da operação
    end = time.time() - begin
    return end
    #return total

tempos = []
for n in range(4):
    N_PROCS = n + 1
    tempo = parallelGA(100, 25, 100, info=False)
    tempos.append(tempo)

plt.plot(tempos)
plt.ylabel('Tempo')
plt.xlabel('Número de Processos')
plt.show()
