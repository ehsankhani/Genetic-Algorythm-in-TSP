import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def tour_distance(tour, cities):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance(cities[tour[i]], cities[tour[i + 1]])
    total_distance += distance(cities[tour[-1]], cities[tour[0]])
    return total_distance

def initialize_population(population_size, num_cities):
    return [np.random.permutation(range(num_cities)) for _ in range(population_size)]

def tournament_selection(population, fitness, team_size):
    selected_indices = np.random.choice(len(population), size=team_size, replace=False)
    tournament_fitness = [fitness[i] for i in selected_indices]
    return selected_indices[np.argmin(tournament_fitness)]

def crossover(parent1, parent2):
    start, end = np.sort(np.random.choice(len(parent1), size=2, replace=False))
    child = [-1] * len(parent1)
    child[start:end + 1] = parent1[start:end + 1]
    remaining_indices = [i for i in range(len(parent1)) if parent2[i] not in child]
    remaining_values = [parent2[i] for i in range(len(parent1)) if parent2[i] not in child]
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining_values.pop(0)
    return child

def mutate(individual):
    idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def genetic_algorithm(num_cities, population_size, team_size, crossover_rate, mutation_rate, elitism_rate, num_generations, output_text, canvas_frame):
    cities = np.random.randint(1, 11, size=(num_cities, 2))  # Use whole numbers between 1 and 10 for coordinates
    city_names = [str(i) for i in range(1, num_cities + 1)]  # Use integers for city names

    population = initialize_population(population_size, num_cities)

    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(num_generations):
        # Calculate fitness as the inverse of tour distance
        fitness = [1 / tour_distance(ind, cities) for ind in population]

        best_fitness = max(fitness)
        avg_fitness = np.mean(fitness)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        num_elites = int(elitism_rate * population_size)
        elite_indices = np.argsort(fitness)[-num_elites:]
        elites = [population[i] for i in elite_indices]

        new_population = elites.copy()

        while len(new_population) < population_size:
            parent1_idx = tournament_selection(population, fitness, team_size)
            parent2_idx = tournament_selection(population, fitness, team_size)

            if np.random.rand() < crossover_rate:
                child = crossover(population[parent1_idx], population[parent2_idx])
            else:
                child = population[parent1_idx].copy()  # Ensure the parent remains unchanged

            if np.random.rand() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population

    final_fitness = [1 / tour_distance(ind, cities) for ind in population]
    best_individual = population[np.argmax(final_fitness)]
    best_tour = [cities[i] for i in best_individual]

    # Print the best tour path with city names and coordinates to the output text
    output_text.insert(END, "Best Tour Path:\n")
    for city_index in best_individual:
        output_text.insert(END, f"{city_names[city_index]}: {cities[city_index]}\n")

    # Plot fitness history
    fig, ax = plt.subplots()
    ax.plot(range(num_generations), avg_fitness_history, label='Average Fitness')
    ax.plot(range(num_generations), best_fitness_history, label='Best Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()

    # Embed the fitness history plot in the GUI
    canvas_fitness = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_fitness_widget = canvas_fitness.get_tk_widget()
    canvas_fitness_widget.pack(side=LEFT, padx=10)

    # Plot the best tour
    best_tour.append(best_tour[0])
    best_tour = np.array(best_tour)
    fig_tour, ax_tour = plt.subplots()
    ax_tour.plot(best_tour[:, 0], best_tour[:, 1], marker='o', linestyle='-')
    ax_tour.set_title('Best Tour')
    ax_tour.set_xlabel('X-coordinate')
    ax_tour.set_ylabel('Y-coordinate')

    # Embed the best tour plot in the GUI
    canvas_tour = FigureCanvasTkAgg(fig_tour, master=canvas_frame)
    canvas_tour_widget = canvas_tour.get_tk_widget()
    canvas_tour_widget.pack(side=LEFT, padx=10)

    return best_tour

def generate(entry_num_cities, entry_population_size, entry_team_size, entry_crossover_rate, entry_mutation_rate, entry_elitism_rate, entry_num_generations, output_text, canvas_frame):
    # Parameters
    num_cities = int(entry_num_cities.get())
    population_size = int(entry_population_size.get())
    team_size = int(entry_team_size.get())
    crossover_rate = float(entry_crossover_rate.get())
    mutation_rate = float(entry_mutation_rate.get())
    elitism_rate = float(entry_elitism_rate.get())
    num_generations = int(entry_num_generations.get())

    # Run the genetic algorithm
    best_tour = genetic_algorithm(num_cities, population_size, team_size, crossover_rate, mutation_rate, elitism_rate, num_generations, output_text, canvas_frame)

# GUI setup
root = Tk()
root.title("Genetic Algorithm for TSP")

# Load and resize the background image
bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((1550, 872))  # Adjust the size as needed
background_image = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Input widgets
label_num_cities = Label(root, text="Number of Cities:")
label_num_cities.grid(row=0, column=0, padx=5, pady=5)
entry_num_cities = Entry(root)
entry_num_cities.insert(0,"10")
entry_num_cities.grid(row=0, column=1, padx=5, pady=5)

label_population_size = Label(root, text="Population Size:")
label_population_size.grid(row=1, column=0, padx=5, pady=5)
entry_population_size = Entry(root)
entry_population_size.insert(0,"100")
entry_population_size.grid(row=1, column=1, padx=5, pady=5)

label_team_size = Label(root, text="Team Size:")
label_team_size.grid(row=2, column=0, padx=5, pady=5)
entry_team_size = Entry(root)
entry_team_size.insert(0, "2")
entry_team_size.grid(row=2, column=1, padx=5, pady=5)

label_crossover_rate = Label(root, text="Crossover Rate:")
label_crossover_rate.grid(row=3, column=0, padx=5, pady=5)
entry_crossover_rate = Entry(root)
entry_crossover_rate.insert(0, "0.75")
entry_crossover_rate.grid(row=3, column=1, padx=5, pady=5)

label_mutation_rate = Label(root, text="Mutation Rate:")
label_mutation_rate.grid(row=4, column=0, padx=5, pady=5)
entry_mutation_rate = Entry(root)
entry_mutation_rate.insert(0, "0.2")
entry_mutation_rate.grid(row=4, column=1, padx=5, pady=5)

label_elitism_rate = Label(root, text="Elitism Rate:")
label_elitism_rate.grid(row=5, column=0, padx=5, pady=5)
entry_elitism_rate = Entry(root)
entry_elitism_rate.insert(0, "0.02")
entry_elitism_rate.grid(row=5, column=1, padx=5, pady=5)

label_num_generations = Label(root, text="Number of Generations:")
label_num_generations.grid(row=6, column=0, padx=5, pady=5)
entry_num_generations = Entry(root)
entry_num_generations.insert(0,"150")
entry_num_generations.grid(row=6, column=1, padx=5, pady=5)

# Output text widget
output_text = Text(root, height=10, width=30)
output_text.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

# Canvas frame to embed plots
canvas_frame = Frame(root)
canvas_frame.grid(row=0, column=2, rowspan=8, padx=10, pady=5)

# Generate button
generate_button = Button(root, text="Generate", command=lambda: generate(entry_num_cities, entry_population_size, entry_team_size, entry_crossover_rate, entry_mutation_rate, entry_elitism_rate, entry_num_generations, output_text, canvas_frame))
generate_button.grid(row=8, column=0, columnspan=2, pady=6)

# Start the GUI main loop
root.mainloop()
