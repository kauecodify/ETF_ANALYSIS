# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: k
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib

# =============================================================================

# =============================================================================
# Configurar o backend do matplotlib para Qt5 antes de importar pyplot
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
                             QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QLabel, QSplitter,
                             QTabWidget, QGroupBox, QFormLayout, QLineEdit, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt
from pmdarima import auto_arima
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class DataFrameTable(QTableWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.load_data()
        
    def load_data(self):
        self.setRowCount(self.df.shape[0])
        self.setColumnCount(self.df.shape[1])
        self.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(self.df.shape[0]):
            for j in range(self.df.shape[1]):
                self.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))
                
        self.resizeColumnsToContents()
# =============================================================================

# =============================================================================

class ARIMA_Perceptron_Analyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETF Analyzer - ARIMA + Perceptron")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variáveis principais
        self.df = None
        self.analysis_results = {}
        self.current_plots = []
        
        # Widgets principais
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Cria abas
        self.data_tab = self.create_data_tab()
        self.analysis_tab = self.create_analysis_tab()
        self.results_tab = self.create_results_tab()
        
        self.tabs.addTab(self.data_tab, "📂 Dados")
        self.tabs.addTab(self.analysis_tab, "🧠 Análise")
        self.tabs.addTab(self.results_tab, "📊 Resultados")
        
        # Conecta sinais
        self.btn_load = None  # Será definido em create_data_tab
        self.btn_run_analysis = None  # Será definido em create_analysis_tab
        self.init_ui()
        
    def init_ui(self):
        # Conecta os sinais após a criação dos botões
        if hasattr(self, 'btn_load') and self.btn_load:
            self.btn_load.clicked.connect(self.load_file)
        if hasattr(self, 'btn_run_analysis') and self.btn_run_analysis:
            self.btn_run_analysis.clicked.connect(self.run_analysis)
        
    def create_data_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controles superiores
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("Carregar XLSX")
        self.file_label = QLabel("Nenhum arquivo carregado")
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.file_label)
        top_layout.addStretch()
        
        layout.addLayout(top_layout)
        
        # Tabela de dados
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        return tab
# =============================================================================

# =============================================================================
        
    def create_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controles de configuração
        config_group = QGroupBox("Configurações de Análise")
        config_layout = QFormLayout(config_group)
        
        self.cb_target = QComboBox()
        self.cb_group = QComboBox()
        self.cb_date = QComboBox()
        self.forecast_input = QLineEdit("30")
        self.epochs_input = QLineEdit("1000")
        self.chk_show_table = QCheckBox("Mostrar tabela de dados")
        self.chk_show_table.setChecked(True)
        
        config_layout.addRow("Coluna Alvo:", self.cb_target)
        config_layout.addRow("Coluna de Grupo:", self.cb_group)
        config_layout.addRow("Coluna de Data:", self.cb_date)
        config_layout.addRow("Dias para Previsão:", self.forecast_input)
        config_layout.addRow("Épocas de Treinamento:", self.epochs_input)
        config_layout.addRow(self.chk_show_table)
        
        layout.addWidget(config_group)
        
        # Botão de execução
        self.btn_run_analysis = QPushButton("Executar Análise ARIMA + Perceptron")
        layout.addWidget(self.btn_run_analysis)
        
        # Área de status
        self.status_label = QLabel("Pronto para analisar")
        self.status_label.setStyleSheet("font-weight: bold; color: #333; padding: 10px;")
        layout.addWidget(self.status_label)
        
        return tab
        
    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Splitter para múltiplos gráficos
        self.splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.splitter)
        
        return tab
# =============================================================================

# =============================================================================
        
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo XLSX", "", "Excel Files (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                self.file_label.setText(os.path.basename(file_path))
                self.df = pd.read_excel(file_path)
                # Substitui a tabela atual pela nova
                self.table = DataFrameTable(self.df)
                # Remove a tabela antiga e adiciona a nova
                for i in reversed(range(self.data_tab.layout().count())):
                    widget = self.data_tab.layout().itemAt(i).widget()
                    if widget is not None and isinstance(widget, QTableWidget):
                        widget.setParent(None)
                self.data_tab.layout().addWidget(self.table)
                
                # Atualiza comboboxes
                self.update_comboboxes()
                self.status_label.setText("Dados carregados com sucesso!")
                self.status_label.setStyleSheet("color: green;")
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar arquivo:\n{str(e)}")
                self.status_label.setText("Erro ao carregar arquivo")
                self.status_label.setStyleSheet("color: red;")
    
    def update_comboboxes(self):
        if self.df is not None:
            columns = self.df.columns.tolist()
            
            self.cb_target.clear()
            self.cb_group.clear()
            self.cb_date.clear()
            
            self.cb_target.addItems(columns)
            self.cb_group.addItems(columns)
            self.cb_date.addItems(columns)
            
            # Tenta identificar automaticamente colunas de data
            date_cols = [col for col in columns if pd.api.types.is_datetime64_any_dtype(self.df[col]) or 'date' in col.lower() or 'data' in col.lower()]
            if date_cols:
                self.cb_date.setCurrentText(date_cols[0])
    
    def run_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, "Aviso", "Carregue um arquivo XLSX primeiro!")
            return
            
        try:
            # Limpa resultados anteriores
            self.clear_plots()
            self.analysis_results = {}
            
            # Obtém parâmetros da interface
            target_col = self.cb_target.currentText()
            group_col = self.cb_group.currentText()
            date_col = self.cb_date.currentText()
            forecast_steps = int(self.forecast_input.text())
            epochs = int(self.epochs_input.text())
            show_table = self.chk_show_table.isChecked()
            
            # Verifica se temos coluna de data
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                except Exception as e:
                    QMessageBox.critical(self, "Erro", f"Falha ao converter coluna de data: {str(e)}")
                    self.status_label.setText("Erro na conversão de data")
                    self.status_label.setStyleSheet("color: red;")
                    return
            
            # Configura índice temporal
            temp_df = self.df.set_index(date_col).sort_index()
            
            # Processa cada grupo separadamente
            groups = temp_df[group_col].unique()
            self.status_label.setText(f"Processando {len(groups)} grupos...")
            self.status_label.setStyleSheet("color: blue;")
            QApplication.processEvents()
            
            for group in groups:
                group_df = temp_df[temp_df[group_col] == group][[target_col]].dropna()
                
                if len(group_df) < 30:
                    self.status_label.setText(f"Grupo {group} ignorado (dados insuficientes)")
                    continue
                
                # Realiza análise
                self.analyze_group(group, group_df, target_col, forecast_steps, epochs, show_table)
                
            # Atualiza interface com resultados
            self.display_results()
            self.status_label.setText("Análise concluída com sucesso!")
            self.status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro na Análise", f"Ocorreu um erro:\n{str(e)}")
            self.status_label.setText("Erro na análise")
            self.status_label.setStyleSheet("color: red;")
# =============================================================================
# =============================================================================
    
    def analyze_group(self, group_name, group_df, target_col, forecast_steps, epochs, show_table):
        """Executa análise ARIMA + Perceptron para um grupo específico"""
        results = {}
        
        # 1. Modelagem ARIMA
        self.status_label.setText(f"Processando ARIMA para {group_name}...")
        QApplication.processEvents()
        
        # Treina modelo ARIMA
        try:
            arima_model = auto_arima(
                group_df[target_col],
                seasonal=False,
                suppress_warnings=True,
                stepwise=True,
                trace=False
            )
            
            # Faz previsão
            arima_forecast, conf_int = arima_model.predict(
                n_periods=forecast_steps,
                return_conf_int=True
            )
            
            # Armazena resultados ARIMA
            results['arima'] = {
                'model': arima_model,
                'forecast': arima_forecast,
                'conf_int': conf_int,
                'last_value': group_df[target_col].iloc[-1]
            }
        except Exception as e:
            self.status_label.setText(f"Erro ARIMA em {group_name}: {str(e)}")
            return
        
        # 2. Perceptron com Features de ARIMA
        self.status_label.setText(f"Treinando Perceptron para {group_name}...")
        QApplication.processEvents()
        
        try:
            # Cria features
            features = self.create_features(group_df, target_col, arima_forecast)
            
            if show_table:
                # Mostra tabela de features em uma nova janela
                self.show_features_table(features, group_name)
            
            # Treina modelo Perceptron
            X = features.drop(columns=['target'])
            y = features['target']
            
            # Normaliza dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modelo MLP
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50, 30),
                activation='relu',
                solver='adam',
                max_iter=epochs,
                random_state=42,
                early_stopping=True
            )
            
            # Treinamento (usando todos os dados para simplificar)
            mlp.fit(X_scaled, y)
            
            # Previsão para o último ponto
            last_data = features.iloc[-1:].drop(columns=['target'])
            last_scaled = scaler.transform(last_data)
            mlp_forecast = mlp.predict(last_scaled)[0]
            
            # Armazena resultados Perceptron
            results['perceptron'] = {
                'model': mlp,
                'forecast': mlp_forecast,
                'scaler': scaler,
                'features': features
            }
            
            # Armazena todos os resultados para este grupo
            self.analysis_results[group_name] = {
                'data': group_df,
                'results': results
            }
        except Exception as e:
            self.status_label.setText(f"Erro Perceptron em {group_name}: {str(e)}")
    
    def create_features(self, df, target_col, arima_forecast):
        """Cria features para o modelo Perceptron"""
        features = pd.DataFrame(index=df.index)
        
        # Features básicas
        features['price'] = df[target_col]
        features['returns'] = df[target_col].pct_change()
        features['volatility'] = df[target_col].pct_change().rolling(10).std()
        
        # Features de momentum
        features['ma_20'] = df[target_col].rolling(20).mean()
        features['ma_50'] = df[target_col].rolling(50).mean()
        features['momentum'] = df[target_col] / df[target_col].shift(4) - 1
        
        # Integra previsão ARIMA
        # Como a previsão ARIMA é para o futuro, usamos o último valor disponível para toda a série
        features['arima_forecast'] = arima_forecast[-1]
        features['arima_diff'] = df[target_col] - arima_forecast[-1]
        features['arima_trend'] = np.sign(arima_forecast[-1] - arima_forecast[0])
        
        # Target (retorno futuro)
        features['target'] = features['returns'].shift(-1)
        
        return features.dropna()
    
    def show_features_table(self, features, group_name):
        """Mostra uma tabela com as features usadas no modelo"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Features - {group_name}")
        dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(dialog)
        table = DataFrameTable(features)
        layout.addWidget(table)
        
        dialog.exec_()
# =============================================================================

# =============================================================================

    def display_results(self):
        """Exibe os resultados na interface"""
        if not self.analysis_results:
            return
            
        # Limpa visualizações anteriores
        self.clear_plots()
        
        # Cria um gráfico para cada grupo
        for group_name, group_data in self.analysis_results.items():
            # Cria figura
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plota dados históricos
            data = group_data['data']
            arima_results = group_data['results']['arima']
            perceptron_results = group_data['results']['perceptron']
            
            # Linha histórica
            ax.plot(data.index, data.iloc[:, 0], 'b-', label='Histórico', linewidth=2)
            
            # Previsão ARIMA
            last_date = data.index[-1]
            forecast_dates = pd.date_range(last_date, periods=len(arima_results['forecast'])+1, freq='B')[1:]
            
            ax.plot(forecast_dates, arima_results['forecast'], 'r--', label='Previsão ARIMA', linewidth=2)
            ax.fill_between(
                forecast_dates,
                arima_results['conf_int'][:, 0],
                arima_results['conf_int'][:, 1],
                color='pink', alpha=0.3
            )
            
            # Previsão Perceptron (para o próximo período)
            next_date = forecast_dates[0]
            perceptron_value = arima_results['last_value'] * (1 + perceptron_results['forecast'])
            ax.plot(next_date, perceptron_value, 'go', markersize=10, label='Previsão Perceptron')
            
            # Configurações do gráfico
            ax.set_title(f"Análise: {group_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel("Valor", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adiciona ao splitter
            canvas = FigureCanvas(fig)
            self.splitter.addWidget(canvas)
            self.current_plots.append((fig, canvas))
    
    def clear_plots(self):
        """Remove todos os gráficos atuais"""
        while self.splitter.count() > 0:
            widget = self.splitter.widget(0)
            widget.setParent(None)
            widget.deleteLater()
            
        self.current_plots = []
    
    def closeEvent(self, event):
        """Limpeza ao fechar a aplicação"""
        self.clear_plots()
        event.accept()

if __name__ == "__main__":
    # Verifica se estamos executando no Spyder
    if 'SPYDER_ARGS' in os.environ:
        print("Executando no Spyder - configurando backend Qt5...")
        matplotlib.use('Qt5Agg')
    
    app = QApplication(sys.argv)
    window = ARIMA_Perceptron_Analyzer()
    window.show()
    app.exec_()
# =============================================================================
