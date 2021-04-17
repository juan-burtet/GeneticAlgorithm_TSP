import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    

    '''
    Calculate the fitness from the route,
    Fitness is the inverse from the distance
    '''
    def routeFitness(self):

        # If fitness wasn't calculated
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        
        return self.fitness

'''
Create a random route between the cities
'''
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

'''
Create a initial population of random routes
'''
def initialPopulation(popSize, cityList):
    population = []

    for i in range(popSize):
        population.append(createRoute(cityList))
    
    return population

'''
Rank the Routes from the best to worst - SERIAL
'''
def rankRoutes(population):
    fitnessResults = {}

    # Calculate the fitness from every route
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

'''
Does the Selection step from a Genetic Algorithm - PARALLEL
Returns the best from the population
'''
def selection(popRanked, eliteSize):
    selectionResults = []

    # Calculate the percent fitness of every route
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum/df.Fitness.sum()

    # Add the best from the population
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
    
    # Uses probability to select a route
    for i in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()

        for i in range(len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    
    return selectionResults

'''
Get the routes that gonna be in the crossover process - PARALLEL
'''
def matingPool(population, selectionResults):
    matingpool = []

    # Add the the selected ones
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    
    return matingpool

'''
Crossover 2 routes in one
'''
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    # Get the interval that gonna be changed
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    # start in the min and ends in the max
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    # Get the route in that interval
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    
    # Add the rest from the other parent
    childP2 = [item for item in parent2 if item not in childP1]

    return childP1 + childP2

'''
Crossover all the matingpool - PARALLEL
'''
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    # Puts the best in the next children
    for i in range(eliteSize):
        children.append(matingpool[i])

    # The next will be every combination from the matingpool
    for i in range(length):
        child = breed(pool[i], pool[len(matingpool) -i -1])
        children.append(child)
    
    return children

'''
Mutate every individual with a mutation rate.
'''
def mutate(individual, mutationRate):

    # check every city
    for swapped in range(len(individual)):

        # Swap this city with a random one
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    
    return individual

'''
Mutate every individual on the population with a mutation rate
'''
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    # Mutate every invidual from a population
    for ind in range(len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    
    return mutatedPop

'''
Create the next generation from the actual gen
'''
def nextGeneration(currentGen, eliteSize, mutationRate):
    
    # Rank the best routes from the population
    popRanked = rankRoutes(currentGen) # SERIAL

    # Select the best routes from the population
    selectionResults = selection(popRanked, eliteSize) # PARALLEL

    # Get the Mating Pool from the population
    matingpool = matingPool(currentGen, selectionResults) # PARALLEL

    # Crossover the mating pool to create the next generation
    children = breedPopulation(matingpool, eliteSize) # PARALLEL
    print(children)

    # Mutate the individuals from the next generation
    nextgeneration = mutatePopulation(children, mutationRate) #

    return nextgeneration


# PARTE PARALELA

# Inicializa a população em paralelo
def parallelInitialPopulation(popSize, cityList, n_procs):
    q_pop = Queue()

    # Divide o tamanho de cada população
    proc_size = int(popSize/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(popSize % n_procs):
        proc_sizes[i] += 1

    # Um processo cria uma quantidade certa de populações
    def proc_init_pop(popSize, cityList, pop):

        # Cria uma certa quantidade de individuos
        population = []
        for i in range(popSize):
            population.append(createRoute(cityList))
        
        # adiciona eles na pop
        pop.put(population)

    # Manda o trabalho pros processos
    procs = []
    for i in range(n_procs):
        p = Process(
            target=proc_init_pop,
            args=(proc_sizes[i], cityList, q_pop)
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    # Esvazia a fila de populações
    pop = []
    while not q_pop.empty():
        pop += q_pop.get()

    # Retorna a população
    return pop

# Ranqueia as populações de maneira paralela
def rankRoutesParallel(population, n_procs):
    fitnessResults = {}

    # Usado para cada processo
    def proc_rank_routes(pop, lock, fit_queue, begin, end):

        # Percorre suas posições
        for i in range(begin, end):

            lock.acquire() # trava essa área

            # Adiciona esse valor no fitnessResults
            fit_res = fit_queue.get()
            fit_res[i] = Fitness(population[i]).routeFitness()
            fit_queue.put(fit_res)

            lock.release() # libera essa área

    # Necessário para o processo
    lock = Lock()
    fit_queue = Queue()
    fit_queue.put(fitnessResults)

    # Divide o tamanho de cada população
    proc_size = int(len(population)/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(len(population) % n_procs):
        proc_sizes[i] += 1
    
    values = []
    for i in range(n_procs):
        
        # Pega os valores iniciais
        if i == 0:
            begin = 0
        else:
            begin = values[i-1][1]
        
        # pega os valores finais
        end = begin + proc_sizes[i]

        values.append((begin, end))

    # Manda o trabalho pros processos
    procs = []
    for i in range(n_procs):
        p = Process(
            target=proc_rank_routes,
            args=(
                population, 
                lock,
                fit_queue,
                values[i][0],
                values[i][1],
            )
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    # Recupera os resultados de Fitness
    fitnessResults = fit_queue.get()
    
    # Retorna eles ordenados
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

# Seleciona os melhores da população de maneira paralela
def selectionParallel(popRanked, eliteSize, n_procs):
    selectionResults = []

    # Calcula a porcentagem de fitness de cada rota
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum/df.Fitness.sum()

    # Guarda os melhores
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])

    # Seleção aleatória feita por cada processo
    def proc_select(popRanked, df, select_queue, size):

        # Usa a probabilidade de pegar uma rota
        for i in range(size):
            pick = 100 * random.random()

            for i in range(len(popRanked)):
                if pick <= df.iat[i,3]:
                    select_queue.put(popRanked[i][0])
                    break

    # Divide o tamanho de cada população
    size = len(popRanked) - eliteSize
    proc_size = int(size/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(size % n_procs):
        proc_sizes[i] += 1

    # Fila de Seleção
    select_queue = Queue()

    # Manda o trabalho pros processos
    procs = []
    for i in range(n_procs):
        p = Process(
            target=proc_select,
            args=(
                popRanked, 
                df, 
                select_queue,
                proc_sizes[i],
            )
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    while not select_queue.empty():
        selectionResults.append(select_queue.get())

    # Retorna a população
    return selectionResults

# Pega os selecionados para serem cruzados
def matingPoolParallel(population, selectionResults, n_procs):
    matingpool = []

    # Divide o tamanho de cada população
    proc_size = int(len(selectionResults)/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(len(selectionResults) % n_procs):
        proc_sizes[i] += 1
    
    # Pega os valores de intervalos de cada um
    values = []
    for i in range(n_procs):
        
        # Pega os valores iniciais
        if i == 0:
            begin = 0
        else:
            begin = values[i-1][1]
        
        # pega os valores finais
        end = begin + proc_sizes[i]

        values.append((begin, end))
    
    # Usado para recuperar os mating pools
    def proc_mat_pool(population, selectionResults, begin, end, mat_queue):

        for i in range(begin, end):
            index = selectionResults[i]
            mat_queue.put(population[index])

    # Manda o trabalho pros processos
    procs = []
    mat_queue = Queue()
    for i in range(n_procs):
        p = Process(
            target=proc_mat_pool,
            args=(
                population, 
                selectionResults,
                values[i][0],
                values[i][1],
                mat_queue,
            )
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    # Pega todos os valores da fila
    while not mat_queue.empty():
        matingpool.append(mat_queue.get())

    # Retorna o mating pool
    return matingpool

# Cruza os selecionados
def breedPopulationParallel(matingpool, eliteSize, n_procs):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    
    # Adiciona os melhores como filhos
    for i in range(eliteSize):
        children.append(matingpool[i])
    
    # Divide o tamanho de cada população
    proc_size = int(length/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(length % n_procs):
        proc_sizes[i] += 1
    
    # Pega os valores de intervalos de cada um
    values = []
    for i in range(n_procs):
        
        # Pega os valores iniciais
        if i == 0:
            begin = 0
        else:
            begin = values[i-1][1]
        
        # pega os valores finais
        end = begin + proc_sizes[i]

        values.append((begin, end))
    
    # cruza os melhores de um certo intervalo
    def proc_breed_pop(size, pool, begin, end, breed_queue):

        for i in range(begin, end):
            child = breed(pool[i], pool[size - i - 1])
            breed_queue.put(child)
    
    # Manda o trabalho pros processos
    procs = []
    breed_queue = Queue()
    for i in range(n_procs):
        p = Process(
            target=proc_breed_pop,
            args=(
                len(matingpool), 
                pool,
                values[i][0],
                values[i][1],
                breed_queue,
            )
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    # Pega todos os valores da fila
    while not breed_queue.empty():
        children.append(breed_queue.get())

    # Retorna as crianças
    return children

# Faz a mutação da nova geração
def mutatePopulationParallel(population, mutationRate, n_procs):
    mutatedPop = []

    # Divide o tamanho de cada população
    proc_size = int(len(population)/n_procs)
    proc_sizes = [proc_size for n in range(n_procs)]
    for i in range(len(population) % n_procs):
        proc_sizes[i] += 1
    
    # Pega os valores de intervalos de cada um
    values = []
    for i in range(n_procs):
        
        # Pega os valores iniciais
        if i == 0:
            begin = 0
        else:
            begin = values[i-1][1]
        
        # pega os valores finais
        end = begin + proc_sizes[i]

        values.append((begin, end))

    # Muta uma parcela da população
    def proc_mutate(population, mutationRate, begin, end, mut_queue):
        # Mutate every invidual from a population
        for ind in range(begin, end):
            mutatedInd = mutate(population[ind], mutationRate)
            mut_queue.put(mutatedInd)

    # Manda o trabalho pros processos
    procs = []
    mut_queue = Queue()
    for i in range(n_procs):
        p = Process(
            target=proc_mutate,
            args=(
                population, 
                mutationRate,
                values[i][0],
                values[i][1],
                mut_queue,
            )
        )
        procs.append(p)
        p.start()
    
    # Finaliza os processos
    for p in procs:
        p.join()

    # Pega todos os valores da fila
    while not mut_queue.empty():
        mutatedPop.append(mut_queue.get())

    # Retorna as crianças
    return mutatedPop

# Faz a criação da nova geração de maneira paralela
def nextGenParallel(gen, eliteSize, mutationRate, n_procs):

    # Ranqueia as melhores populações
    #popRanked = rankRoutesParallel(gen, n_procs)
    popRanked = rankRoutes(gen)

    # Seleciona os melhores da população
    selectionResults = selectionParallel(popRanked, eliteSize, n_procs) # PARALLEL

    # Adiciona os selecionados para cruzamento
    matingpool = matingPoolParallel(gen, selectionResults, n_procs) # PARALLEL

    # Cruza os selecionados para gerar os filhos
    children = breedPopulationParallel(matingpool, eliteSize, n_procs) # PARALLEL
    print(children)
    
    # Mutação dos individuos pra próxima população
    nextgeneration = mutatePopulationParallel(children, mutationRate, n_procs) # PARALLEL

    return nextgeneration

def geneticAlgorithmParallel(population, popSize, eliteSize, mutationRate, generations, n_procs):
    
    # Cria a população inicial
    pop = parallelInitialPopulation(popSize, population, n_procs)

    # Adiciona o progresso
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    # Percorre pela quantidade de populações
    for i in tqdm(range(generations)):
        pop = nextGenParallel(pop, eliteSize, mutationRate, n_procs)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# Sequencial

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):

    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in tqdm(range(generations)):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        break
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in tqdm(range(generations)):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()




CITY_SIZE = 25
POP_SIZE = 100
ELITE_SIZE = 20
MUTATION_RATE = 0.01
GENERATIONS = 500


cityList = []
for i in range(CITY_SIZE):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

#geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
geneticAlgorithmParallel(
    population=cityList, 
    popSize=POP_SIZE, 
    eliteSize=ELITE_SIZE, 
    mutationRate=MUTATION_RATE, 
    generations=GENERATIONS,
    n_procs=5
)

