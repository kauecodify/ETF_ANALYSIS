# ETF_ANALYSIS
ETF Analyzer: ARIMA + Perceptron

---

📌 Descrição
Esta aplicação PyQt5 combina modelos ARIMA (para previsão de séries temporais) com Perceptron Multicamadas (MLP) para analisar dados de ETFs. Ela permite carregar arquivos Excel, selecionar colunas para análise e visualizar previsões híbridas em uma interface gráfica intuitiva.

---

✅ Funcionalidades Principais
Importação de Dados:

Carrega arquivos Excel (.xlsx/.xls)

Visualização tabular dos dados

Detecção automática de colunas de data

Configuração Flexível:

Seleção de coluna alvo, grupo e data

Ajuste de dias para previsão e épocas de treinamento

Opção para visualizar tabela de features

#### **Aba "Resultados"**

- Após a análise, os gráficos serão exibidos um abaixo do outro (um por grupo).

- **Cada gráfico contém**:

- **Linha Azul**: Dados históricos.

- **Linha Vermelha Tracejada**: Previsão ARIMA para o futuro, com intervalo de confiança em rosa.

- **Ponto Verde**: Previsão do Perceptron para o próximo período.

#### **Interpretando os Resultados**

- **ARIMA**: Modelo estatístico para séries temporais. A previsão é uma extrapolação baseada em padrões históricos.

- **Perceptron**: Rede neural que usa features derivadas dos dados e da previsão ARIMA para prever o retorno do próximo período.

#### **Recursos Adicionais**

- **Tabela de Features**: Se marcada, uma janela mostrará as features usadas pelo Perceptron (para entender o que o modelo considerou).

- **Zoom e Navegação**: Use os controles do matplotlib para zoom e pan nos gráficos.

### Exemplo de Dados:

Seu arquivo Excel deve ter pelo menos:

- Uma coluna de datas (reconhecível como data)

- Uma coluna numérica (valores a serem previstos)

- Uma coluna de grupos (para segmentação)

---

Análise Híbrida:

ARIMA Automático:

Modelagem automática com pmdarima

Previsões com intervalos de confiança

---

Perceptron Multicamadas:

Geração automática de features (retornos, volatilidade, momentum)

Integração com previsões do ARIMA

Normalização dos dados e treinamento com early stopping

---

Visualização:

Gráficos comparativos (histórico vs previsões)

Destaque para previsões do perceptron

Tabelas interativas de features

---

⚙️ Requisitos de Instalação

bash
pip install pandas numpy matplotlib pmdarima scikit-learn PyQt5 xlrd openpyxl

---

🚀 Como Executar
Execute o script Python diretamente:


bash
python etf_analyzer.py
🧩 Estrutura do Código

DataFrameTable:

Widget personalizado para exibição de DataFrames
ARIMA_Perceptron_Analyzer (QMainWindow):

Guias:

Dados: Importação e visualização
Análise: Configuração dos modelos
Resultados: Visualização gráfica

Fluxo de Análise:

Diagram Code
graph TD
A[Carregar Excel] --> B[Selecionar Colunas]
B --> C[Treinar ARIMA]
C --> D[Gerar Features]
D --> E[Treinar Perceptron]
E --> F[Plotar Resultados]

---

⚠️ Notas Importantes

Para execução no Spyder: Ativa automaticamente o backend Qt5Agg

Dados mínimos: 30 observações por grupo

Colunas de data são detectadas automaticamente (ex: "Date", "Data")


mit license +_*

by k

![imagem_2025-06-14_170046902-removebg-preview (2)](https://github.com/user-attachments/assets/b2710e24-3acb-4d58-8984-3266ed8657bf)
