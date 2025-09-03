# ## Trabalho da disciplina Redes Complexas
# Aluno: Yago Presto

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
import random
import subprocess
import sys

# Verificar e instalar dependências necessárias
try:
    import scipy
except ImportError:
    print("Instalando scipy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import scipy

try:
    import seaborn
except ImportError:
    print("Instalando seaborn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn

# Carregando o arquivo Excel do dataset Steam
df = pd.read_excel('C:/Users/yagop/Documents/Codes/Alex/Dataset.xlsx', sheet_name='10k+')

# Inicializando o grafo
G = nx.Graph()

# Gerando arestas do grafo a partir do dataframe
for index, row in df.iterrows():
    G.add_edge(row['Titulo'], row['Tag'])

# Função para calcular métricas de uma rede
def calculate_metrics(G):
    if nx.is_connected(G):
        # Coeficiente de clustering médio
        clustering = nx.average_clustering(G)
        # Comprimento médio do caminho (somente para grafos conectados)
        avg_path_length = nx.average_shortest_path_length(G)
        # Grau médio
        degree_mean = sum(dict(G.degree()).values()) / len(G)
        # Diâmetro da rede
        diametro = nx.diameter(G)
    else:
        # Se a rede não for conectada, não podemos calcular o caminho médio ou o diâmetro
        clustering = nx.average_clustering(G)
        avg_path_length = None
        degree_mean = sum(dict(G.degree()).values()) / len(G)
        diametro = None

    return clustering, avg_path_length, degree_mean, diametro

# Calcular métricas para a rede criada a partir do Excel
clustering_g, avg_path_length_g, degree_mean_g, diametro_g = calculate_metrics(G)

# Exibir os resultados para a rede criada pelo Excel
print("Métricas da rede criada:")
if diametro_g is not None:
    print(f"Diâmetro da rede: {diametro_g}")
else:
    print("Diâmetro da rede: Grafo desconectado")

if avg_path_length_g is not None:
    print(f"Comprimento Médio do Caminho: {avg_path_length_g:.4f}")
else:
    print("Comprimento Médio do Caminho: Grafo desconectado")

print(f"Grau Médio: {degree_mean_g:.4f}")

# Calcular Closeness e Betweenness
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Exibir os primeiros valores de Closeness e Betweenness
print("\nCloseness (Top 10):")
for no, centralidade in sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{no}: {centralidade:.4f}")

print("\nBetweenness (Top 10):")
for no, centralidade in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{no}: {centralidade:.4f}")

# Plotar a rede (usando um layout mais simples que não requer scipy)
plt.figure(figsize=(12, 12))
nx.draw(G, with_labels=False, node_size=50, node_color="skyblue", font_size=8, font_weight="bold", pos=nx.random_layout(G))
plt.title("Visualização do Grafo")
plt.show()

# Plotar o histograma dos graus com escala log-log
graus = [grau for no, grau in G.degree()]
plt.figure(figsize=(10, 6))
plt.hist(graus, bins=range(1, max(graus) + 2), color='skyblue', edgecolor='black', log=True)
plt.xscale('log')
plt.yscale('log')
plt.title("Histograma dos Graus dos Nós (Escala Log-Log)")
plt.xlabel("Grau (log)")
plt.ylabel("Frequência (log)")
plt.grid(True)
plt.show()

# Plotar o histograma do closeness
closeness_values = list(closeness_centrality.values())
plt.figure(figsize=(10, 6))
plt.hist(closeness_values, bins=30, color='green', edgecolor='black')
plt.title("Histograma do Closeness Centrality")
plt.xlabel("Closeness")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# Plotar o histograma do betweenness
betweenness_values = list(betweenness_centrality.values())
plt.figure(figsize=(10, 6))
plt.hist(betweenness_values, bins=30, color='purple', edgecolor='black')
plt.title("Histograma do Betweenness Centrality")
plt.xlabel("Betweenness")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# Plotar o gráfico de scatter para Closeness e Betweenness
plt.figure(figsize=(10, 6))
plt.scatter(closeness_values, betweenness_values, color='orange', edgecolors='black', alpha=0.5)
plt.title("Relação entre Closeness e Betweenness")
plt.xlabel("Closeness")
plt.ylabel("Betweenness")
plt.grid(True)
plt.show()

# Calcular o grau de cada nó (número de arestas conectadas a ele)
grau_nos = dict(G.degree())

# Extrair os graus para uma lista
graus = list(grau_nos.values())

# Plotar o histograma de distribuição de grau em escala log-log
plt.figure(figsize=(10, 6))
plt.hist(graus, bins=np.logspace(np.log10(min(graus)), np.log10(max(graus)), num=30), edgecolor='black', alpha=0.75)
plt.xscale('log')
plt.yscale('log')
plt.title("Histograma de Distribuição de Grau (Escala Log-Log)")
plt.xlabel("Grau (log)")
plt.ylabel("Frequência (log)")
plt.grid(True, which="both", ls="--")
plt.show()

# Contar quantos nós possuem cada grau específico
grau_contagem = collections.Counter(grau_nos.values())

# Separar os graus e o número de nós para plotagem
graus = sorted(grau_contagem.keys())
numero_de_enlaces = [grau_contagem[grau] for grau in graus]

# Criar o gráfico de dispersão do grau vs número de enlaces
plt.figure(figsize=(10, 6))
plt.scatter(graus, numero_de_enlaces, color='blue', edgecolor='black', alpha=0.75)
plt.xscale('log')
plt.yscale('log')
plt.title("Grau do Nó vs Número de Enlaces")
plt.xlabel("Grau do Nó (log)")
plt.ylabel("Número de Enlaces (log)")
plt.grid(True, which="both", ls="--")
plt.show()

# Encontrar o componente gigante (maior componente conexo)
largest_cc = max(nx.connected_components(G), key=len)
G_giant = G.subgraph(largest_cc).copy()

# Número de nós e arestas do componente gigante
num_nodes = G_giant.number_of_nodes()
num_edges = G_giant.number_of_edges()

# 1. Calcular o coeficiente de agrupamento (clustering coefficient)
clustering_coefficient = nx.average_clustering(G_giant)
print(f"Coeficiente de Agrupamento (Clustering Coefficient): {clustering_coefficient}")

# 2. Calcular o comprimento médio do caminho (average path length)
avg_path_length = nx.average_shortest_path_length(G_giant)
print(f"Comprimento Médio do Caminho (Average Path Length): {avg_path_length}")

# 4. Comparar com grafos de referência
num_vertices = G.number_of_nodes()
num_arestas_presentes = G.number_of_edges()

# Cálculo do número de arestas possíveis
num_arestas_possiveis = num_vertices * (num_vertices - 1) // 2

# Calcular a razão
razao = num_arestas_presentes / num_arestas_possiveis if num_arestas_possiveis > 0 else 0

# Grafo aleatório Erdős–Rényi
G_random = nx.erdos_renyi_graph(num_nodes, razao)
# Verificar se o grafo aleatório é conexo
if nx.is_connected(G_random):
    random_avg_path_length = nx.average_shortest_path_length(G_random)
else:
    largest_cc_random = max(nx.connected_components(G_random), key=len)
    G_random_giant = G_random.subgraph(largest_cc_random).copy()
    random_avg_path_length = nx.average_shortest_path_length(G_random_giant)

random_clustering = nx.average_clustering(G_random)
print(f"Coeficiente de Agrupamento (Grafo Aleatório): {random_clustering}")
print(f"Comprimento Médio do Caminho (Grafo Aleatório): {random_avg_path_length}")

# Grafo Small-World de Watts-Strogatz
k = int(2 * num_edges / num_nodes)  # Número médio de vizinhos por nó
G_small_world = nx.watts_strogatz_graph(num_nodes, 4, razao)

# Verificar se o grafo Small-World é conexo
if nx.is_connected(G_small_world):
    small_world_avg_path_length = nx.average_shortest_path_length(G_small_world)
else:
    largest_cc_sw = max(nx.connected_components(G_small_world), key=len)
    G_small_world_giant = G_small_world.subgraph(largest_cc_sw).copy()
    small_world_avg_path_length = nx.average_shortest_path_length(G_small_world_giant)

small_world_clustering = nx.average_clustering(G_small_world)
print(f"Coeficiente de Agrupamento (Small-World): {small_world_clustering}")
print(f"Comprimento Médio do Caminho (Small-World): {small_world_avg_path_length}")

# Grafo Preferential Attachment (Barabási-Albert)
m = int(num_edges // num_nodes)  # Número de arestas por novo vértice (aproximado pelo grau médio)
G_barabasi = nx.barabasi_albert_graph(num_nodes, 4)

# Verificar se o grafo Barabási-Albert é conexo
if nx.is_connected(G_barabasi):
    barabasi_avg_path_length = nx.average_shortest_path_length(G_barabasi)
else:
    largest_cc_barabasi = max(nx.connected_components(G_barabasi), key=len)
    G_barabasi_giant = G_barabasi.subgraph(largest_cc_barabasi).copy()
    barabasi_avg_path_length = nx.average_shortest_path_length(G_barabasi_giant)

barabasi_clustering = nx.average_clustering(G_barabasi)
print(f"Coeficiente de Agrupamento (Barabási-Albert): {barabasi_clustering}")
print(f"Comprimento Médio do Caminho (Barabási-Albert): {barabasi_avg_path_length}")

# 5. Comparar as propriedades do grafo real com os modelos
print("\nComparação com os modelos:")
print(f"Coeficiente de Agrupamento do Componente Gigante: {clustering_coefficient}")
print(f"Coeficiente de Agrupamento do Grafo Aleatório: {random_clustering}")
print(f"Coeficiente de Agrupamento do Grafo Small-World: {small_world_clustering}")
print(f"Coeficiente de Agrupamento do Grafo Barabási-Albert: {barabasi_clustering}")

print(f"\nComprimento Médio do Caminho do Componente Gigante: {avg_path_length}")
print(f"Comprimento Médio do Caminho do Grafo Aleatório: {random_avg_path_length}")
print(f"Comprimento Médio do Caminho do Grafo Small-World: {small_world_avg_path_length}")
print(f"Comprimento Médio do Caminho do Grafo Barabási-Albert: {barabasi_avg_path_length}")

# Função para calcular a fração do maior componente conexo
def get_largest_component_fraction(G):
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / G.number_of_nodes()

# Função para simular falhas aleatórias removendo nós aleatoriamente
def random_failure_analysis(G, num_removals):
    G_copy = G.copy()
    initial_largest_component = get_largest_component_fraction(G_copy)
    largest_component_fractions = [initial_largest_component]

    nodes = list(G_copy.nodes())
    random.shuffle(nodes)

    for i in range(num_removals):
        node_to_remove = nodes.pop()
        G_copy.remove_node(node_to_remove)
        largest_component_fractions.append(get_largest_component_fraction(G_copy))

    return largest_component_fractions

# Função para simular ataque removendo nós com maior grau (ou outra medida de centralidade)
def targeted_attack_analysis(G, num_removals, attack_type='degree'):
    G_copy = G.copy()
    initial_largest_component = get_largest_component_fraction(G_copy)
    largest_component_fractions = [initial_largest_component]

    if attack_type == 'degree':
        nodes_by_centrality = sorted(G_copy.degree(), key=lambda x: x[1], reverse=True)
    elif attack_type == 'betweenness':
        centrality = nx.betweenness_centrality(G_copy)
        nodes_by_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Tipo de ataque desconhecido: {attack_type}")

    for i in range(num_removals):
        node_to_remove = nodes_by_centrality.pop(0)[0]
        G_copy.remove_node(node_to_remove)
        largest_component_fractions.append(get_largest_component_fraction(G_copy))

    return largest_component_fractions

# Número de nós a serem removidos
num_removals = 20

# 1. Análise de falha aleatória
random_failures = random_failure_analysis(G, num_removals)

# 2. Análise de ataque direcionado (por grau)
targeted_attacks_degree = targeted_attack_analysis(G, num_removals, attack_type='degree')

# 3. Análise de ataque direcionado (por centralidade de intermediação - betweenness)
targeted_attacks_betweenness = targeted_attack_analysis(G, num_removals, attack_type='betweenness')

# Plotando os resultados
plt.figure(figsize=(10,6))
plt.plot(range(num_removals + 1), random_failures, label="Falha Aleatória", marker='o')
plt.plot(range(num_removals + 1), targeted_attacks_degree, label="Ataque Direcionado (Grau)", marker='s')
plt.plot(range(num_removals + 1), targeted_attacks_betweenness, label="Ataque Direcionado (Betweenness)", marker='^')
plt.xlabel('Número de Nós Removidos')
plt.ylabel('Fração do Maior Componente Conexo')
plt.title('Análise de Ataque e Falha em Grafos')
plt.legend()
plt.grid(True)
plt.show()

# Função para calcular a distância média entre nós no maior componente
def get_average_distance(G):
    if G.number_of_nodes() > 0:  # Verifica se o grafo tem nós
        try:
            largest_cc = max(nx.connected_components(G), key=len)  # Maior componente conexo
            return nx.average_shortest_path_length(G.subgraph(largest_cc))
        except nx.NetworkXError:
            return float('inf')  # Retorna infinito se o grafo não é conexo
    return float('inf')  # Retorna infinito se não há nós

# Função para simular falhas aleatórias removendo nós aleatoriamente
def random_failure_analysis_distance(G, num_removals):
    G_copy = G.copy()
    average_distances = [get_average_distance(G_copy)]  # Distância média inicial

    nodes = list(G_copy.nodes())
    random.shuffle(nodes)

    for i in range(num_removals):
        node_to_remove = nodes.pop()
        G_copy.remove_node(node_to_remove)
        average_distances.append(get_average_distance(G_copy))

    return average_distances

# Função para simular ataque removendo nós com maior grau (ou outra medida de centralidade)
def targeted_attack_analysis_distance(G, num_removals, attack_type='degree'):
    G_copy = G.copy()
    average_distances = [get_average_distance(G_copy)]  # Distância média inicial

    if attack_type == 'degree':
        nodes_by_centrality = sorted(G_copy.degree(), key=lambda x: x[1], reverse=True)
    elif attack_type == 'betweenness':
        centrality = nx.betweenness_centrality(G_copy)
        nodes_by_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Tipo de ataque desconhecido: {attack_type}")

    for i in range(num_removals):
        node_to_remove = nodes_by_centrality.pop(0)[0]
        G_copy.remove_node(node_to_remove)
        average_distances.append(get_average_distance(G_copy))

    return average_distances

# Número de nós a serem removidos
num_removals = 20

# 1. Análise de falha aleatória
random_failures = random_failure_analysis_distance(G, num_removals)

# 2. Análise de ataque direcionado (por grau)
targeted_attacks_degree = targeted_attack_analysis_distance(G, num_removals, attack_type='degree')

# 3. Análise de ataque direcionado (por centralidade de intermediação - betweenness)
targeted_attacks_betweenness = targeted_attack_analysis_distance(G, num_removals, attack_type='betweenness')

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(range(num_removals + 1), random_failures, label="Falha Aleatória", marker='o')
plt.plot(range(num_removals + 1), targeted_attacks_degree, label="Ataque Direcionado (Grau)", marker='s')
plt.plot(range(num_removals + 1), targeted_attacks_betweenness, label="Ataque Direcionado (Betweenness)", marker='^')
plt.xlabel('Número de Nós Removidos')
plt.ylabel('Distância Média entre Nós no Maior Componente')
plt.title('Análise de Ataque e Falha em Grafos')
plt.legend()
plt.grid(True)
plt.ylim(0, 10)  # Define um limite no eixo y para melhor visualização
plt.show()

# Modelo SIR para análise de difusão de informação
def sir_model(G, initial_infected, beta, gamma, num_steps):
    # Estado dos nós: 0 = suscetível, 1 = infectado, 2 = recuperado
    state = {node: 0 for node in G.nodes()}

    # Inicializa os nós infectados
    infected_nodes = random.sample(list(G.nodes()), initial_infected)
    for node in infected_nodes:
        state[node] = 1  # Marcando como infectado

    # Armazenar a proporção de suscetíveis, infectados и recuperados
    susceptible_counts = []
    infected_counts = []
    recovered_counts = []

    for step in range(num_steps):
        new_state = state.copy()
        for node in G.nodes():
            if state[node] == 1:  # Se o nó está infectado
                # Espalha a infecção para vizinhos
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:  # Se o vizinho é suscetível
                        if random.random() < beta:  # Probabilidade de contágio
                            new_state[neighbor] = 1  # Contamina o vizinho
                # O nó recupera com probabilidade gamma
                if random.random() < gamma:
                    new_state[node] = 2  # Marca como recuperado

        # Atualiza o estado
        state = new_state
        susceptible_counts.append(sum(1 for s in state.values() if s == 0))
        infected_counts.append(sum(1 for s in state.values() if s == 1))
        recovered_counts.append(sum(1 for s in state.values() if s == 2))

    return susceptible_counts, infected_counts, recovered_counts

# Parâmetros do modelo
initial_infected = 5
beta = 0.1  # Probabilidade de contágio
gamma = 0.05  # Probabilidade de recuperação
num_steps = 50  # Número de passos na simulação

# Executar o modelo SIR
susceptible_counts, infected_counts, recovered_counts = sir_model(G, initial_infected, beta, gamma, num_steps)

# Plotando os resultados
plt.figure(figsize=(12, 6))
plt.plot(susceptible_counts, label='Suscetíveis', color='blue')
plt.plot(infected_counts, label='Infectados', color='red')
plt.plot(recovered_counts, label='Recuperados', color='green')
plt.xlabel('Passos de Tempo')
plt.ylabel('Número de Nós')
plt.title('Análise de Difusão de Informação usando o Modelo SIR')
plt.legend()
plt.grid(True)
plt.show()

# Função para calcular a distância média entre nós no maior componente
def get_average_distance(G):
    if G.number_of_nodes() > 0:
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            return nx.average_shortest_path_length(G.subgraph(largest_cc))
        except nx.NetworkXError:
            return float('inf')
    return float('inf')

# Função para adicionar e remover nós e arestas dinamicamente
def dynamic_graph_analysis(initial_graph, num_iterations, add_prob=0.5):
    G = initial_graph.copy()
    average_distances = []
    component_sizes = []

    for _ in range(num_iterations):
        # Adiciona um nó
        G.add_node(len(G))

        # Adiciona uma aresta aleatoriamente com probabilidade 'add_prob'
        if random.random() < add_prob and len(G) > 1:
            # Converter nodes para lista antes de usar random.sample
            nodes_list = list(G.nodes())
            node1, node2 = random.sample(nodes_list, 2)
            G.add_edge(node1, node2)

        # Remove um nó aleatoriamente
        if len(G) > 1:
            # Converter nodes para lista antes de usar random.choice
            node_to_remove = random.choice(list(G.nodes()))
            G.remove_node(node_to_remove)

        # Cálculo das propriedades
        average_distance = get_average_distance(G)
        average_distances.append(average_distance)
        largest_component_size = max(len(cc) for cc in nx.connected_components(G))
        component_sizes.append(largest_component_size)

    return average_distances, component_sizes

# Número de iterações para a análise dinâmica
num_iterations = 100

# Executar a análise dinâmica
average_distances, component_sizes = dynamic_graph_analysis(G, num_iterations)

# Verificar se as listas têm o comprimento correto
if len(average_distances) != num_iterations or len(component_sizes) != num_iterations:
    raise ValueError("As listas de resultados devem ter o comprimento igual ao número de iterações.")

# Plotando os resultados
plt.figure(figsize=(12, 6))

# Plot da distância média
plt.subplot(1, 2, 1)
plt.plot(range(num_iterations), average_distances, label="Distância Média", color='blue')
plt.xlabel('Iterações')
plt.ylabel('Distância Média entre Nós')
plt.title('Análise Dinâmica: Distância Média')
plt.grid(True)
plt.legend()

# Plot do tamanho do maior componente
plt.subplot(1, 2, 2)
plt.plot(range(num_iterations), component_sizes, label="Tamanho do Maior Componente", color='orange')
plt.xlabel('Iterações')
plt.ylabel('Tamanho do Maior Componente Conexo')
plt.title('Análise Dinâmica: Tamanho do Maior Componente')
plt.grid(True)
plt.legend()

# Ajustar o layout e mostrar o gráfico
plt.tight_layout()
plt.show()