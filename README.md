Steam Network Analysis 🎮📊
Este repositório contém a análise de redes complexas aplicada ao dataset Steam Games Dataset, disponível no Kaggle:
👉 Steam Games Dataset

O objetivo é explorar como jogos e suas respectivas tags podem ser representados como uma rede, aplicando conceitos de Teoria de Redes, Métricas de Centralidade, Robustez Estrutural e Modelos de Difusão.

📂 Estrutura do Repositório
Dataset.xlsx → Arquivo base contendo os dados da Steam.
SteamDataAnalysis.py → Script em Python com toda a análise de redes complexas.
README.md → Este arquivo de documentação.
🛠️ Dependências
Para executar o projeto, é necessário ter Python 3.8+ instalado.
As bibliotecas utilizadas são:

pandas
numpy
matplotlib
seaborn
networkx
scipy
As dependências são verificadas e instaladas automaticamente pelo script, mas você pode garantir tudo com:

bash pip install -r requirements.txt

📊 Metodologia O projeto segue as seguintes etapas:

Carregamento do Dataset O Excel Dataset.xlsx é carregado, usando a aba "10k+", contendo jogos e suas respectivas tags.

Construção do Grafo Cada jogo é conectado às suas tags, formando um grafo bipartido. Depois, métricas e análises são aplicadas a essa rede.

Métricas de Redes

Grau médio

Diâmetro da rede

Comprimento médio do caminho

Coeficiente de clusterização

Centralidades (Closeness e Betweenness)

Visualizações

Grafo da rede

Histogramas de graus, centralidades e distribuições

Dispersão de Closeness × Betweenness

Componentes e Comparações

Identificação do componente gigante

Comparação com modelos de referência:

Erdős–Rényi (Aleatório)

Watts–Strogatz (Small-World)

Barabási–Albert (Preferential Attachment)

Robustez da Rede

Análise de falhas aleatórias

Análise de ataques direcionados (grau e betweenness)

Impacto na conectividade e distância média da rede

Modelo de Difusão (SIR) Simulação de disseminação de informação (ou epidemias) na rede, com parâmetros ajustáveis de contágio (β) e recuperação (γ).

Análise Dinâmica Estudo do comportamento da rede ao longo de adições e remoções de nós e arestas.

📈 Resultados Esperados Identificar a estrutura da rede de jogos e tags.

Comparar a rede real com modelos teóricos.

Avaliar a resiliência da rede a falhas e ataques.

Simular a propagação de informações/jogadores dentro da rede.
