import random
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Bio import SeqIO, Entrez
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio.Seq import Seq
import re
import threading
import requests
import json
import os
import numpy as np
from urllib.error import HTTPError
from MLOrfScorer import MLOrfScorer
import webbrowser


class GenePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GeneFinder - Herramienta de Predicción de Genes")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")

        #variables para almacenar información
        self.sequence = None
        self.seq_name = ""
        self.sequence_loaded = False
        self.current_frame = None
        self.predictions = []
        self.db_results = []
        self.orf_scorer = MLOrfScorer(model_path="model.joblib")
        #Email para Entrez
        Entrez.email = "ja2958110@gmail.com"

        self.create_widgets()

    def create_widgets(self):
        #panel lateral para opciones
        self.sidebar = tk.Frame(self.root, width=250, bg="#2c3e50", padx=10, pady=10)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        #titulo
        title_label = tk.Label(self.sidebar, text="GeneFinder", font=("Helvetica", 18, "bold"),
                               bg="#2c3e50", fg="white")
        title_label.pack(pady=20)

        #botones del menu lateral
        btn_styles = {'font': ('Helvetica', 11), 'bg': '#34495e', 'fg': 'white',
                      'activebackground': '#3498db', 'activeforeground': 'white',
                      'width': 20, 'height': 2, 'cursor': 'hand2', 'bd': 0}

        self.btn_load = tk.Button(self.sidebar, text="Cargar Secuencia", command=self.load_sequence, **btn_styles)
        self.btn_load.pack(pady=10)

        self.btn_find_orfs = tk.Button(self.sidebar, text="Encontrar ORFs",
                                       command=self.show_orf_finder, state=tk.DISABLED, **btn_styles)
        self.btn_find_orfs.pack(pady=10)

        self.btn_visualize = tk.Button(self.sidebar, text="Visualizar Secuencia",
                                       command=self.show_sequence_view, state=tk.DISABLED, **btn_styles)
        self.btn_visualize.pack(pady=10)

        self.btn_validate = tk.Button(self.sidebar, text="Validar Predicciones",
                                      command=self.show_validation, state=tk.DISABLED, **btn_styles)
        self.btn_validate.pack(pady=10)

        self.btn_compare_scores = tk.Button(self.sidebar, text="Comparar Scores",
                                            command=self.show_score_comparison, state=tk.DISABLED, **btn_styles)
        self.btn_compare_scores.pack(pady=10)

        self.btn_score_orfs = tk.Button(self.sidebar, text="Puntuar ORFs",
                                        command=self.score_orfs, state=tk.DISABLED, **btn_styles)
        self.btn_score_orfs.pack(pady=10)
        
        self.btn_smorf = tk.Button(self.sidebar, text="Analizar smORF",
                           command=self.show_smorf_analysis, **btn_styles)
        self.btn_smorf.pack(pady=10)

        self.btn_about = tk.Button(self.sidebar, text="Acerca de", command=self.show_about, **btn_styles)
        self.btn_about.pack(pady=10)

        #separador
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', pady=20)

        #informacion de la secuencia cargada
        self.sequence_info_frame = tk.Frame(self.sidebar, bg="#2c3e50")
        self.sequence_info_frame.pack(pady=10, fill='x')

        self.seq_info_label = tk.Label(self.sequence_info_frame, text="No hay secuencia cargada",
                                       font=("Helvetica", 10), bg="#2c3e50", fg="white",
                                       wraplength=230, justify='left')
        self.seq_info_label.pack(pady=5)

        #area principal de contenido
        self.main_content = tk.Frame(self.root, bg="#ecf0f1")
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        #mostrar la pantalla de bienvenida
        self.show_welcome()

    def show_welcome(self):
        self.clear_main_content()

        welcome_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=30, pady=30)
        welcome_frame.pack(fill=tk.BOTH, expand=True)

        #titulo de bienvenida
        welcome_title = tk.Label(welcome_frame, text="Bienvenido a GeneFinder",
                                 font=("Helvetica", 24, "bold"), bg="#ecf0f1")
        welcome_title.pack(pady=20)

        #descritpcion
        description = """
        GeneFinder es una herramienta para la predicción de genes en secuencias genómicas.

        Esta aplicación te permite:
        • Cargar secuencias genómicas en formato FASTA
        • Detectar marcos de lectura abiertos (ORFs)
        • Visualizar características de la secuencia
        • Validar tus predicciones con bases de datos externas

        Para comenzar, haz clic en "Cargar Secuencia".
        """

        desc_label = tk.Label(welcome_frame, text=description, font=("Helvetica", 12),
                              bg="#ecf0f1", justify=tk.LEFT, wraplength=700)
        desc_label.pack(pady=20)

        #boton para comenzar
        start_btn = tk.Button(welcome_frame, text="Cargar Secuencia", font=("Helvetica", 12, "bold"),
                              bg="#2980b9", fg="white", padx=15, pady=8, bd=0,
                              command=self.load_sequence, cursor="hand2")
        start_btn.pack(pady=30)

    def load_sequence(self):
        #abrir file dialog seleccionar archivo FASTA
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo FASTA",
            filetypes=[("Archivos FASTA", "*.fasta *.fa"), ("Todos los archivos", "*.*")]
        )

        if not file_path:
            return

        try:
            #cargar la secuencia
            record = list(SeqIO.parse(file_path, "fasta"))[0]
            self.sequence = record.seq
            self.seq_name = record.id
            self.sequence_loaded = True

            #activar botones
            self.btn_find_orfs.config(state=tk.NORMAL)
            self.btn_visualize.config(state=tk.NORMAL)
            self.btn_validate.config(state=tk.NORMAL)
            self.btn_score_orfs.config(state=tk.NORMAL)
            self.btn_compare_scores.config(state=tk.NORMAL)

            #actualizar info de la secuencia
            seq_info = f"Secuencia: {self.seq_name}\nLongitud: {len(self.sequence)} bp"
            self.seq_info_label.config(text=seq_info)

            #mostrar la pantalla de visualización de secuencia
            self.show_sequence_view()

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la secuencia: {str(e)}")

    def clear_main_content(self):
        #limpia screen de contenido principal
        for widget in self.main_content.winfo_children():
            widget.destroy()

    def show_sequence_view(self):
        if not self.sequence_loaded:
            messagebox.showwarning("Advertencia", "Primero debes cargar una secuencia.")
            return

        self.clear_main_content()

        #frame para la visualización de secuencia
        seq_view_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        seq_view_frame.pack(fill=tk.BOTH, expand=True)

        #titulo
        title = tk.Label(seq_view_frame, text="Visualización de Secuencia",
                         font=("Helvetica", 18, "bold"), bg="#ecf0f1")
        title.pack(pady=10)

        #info de la secuencia
        info_frame = tk.Frame(seq_view_frame, bg="#ecf0f1")
        info_frame.pack(fill=tk.X, pady=10)

        tk.Label(info_frame, text=f"ID: {self.seq_name}", font=("Helvetica", 12),
                 bg="#ecf0f1", anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, text=f"Longitud: {len(self.sequence)} bp", font=("Helvetica", 12),
                 bg="#ecf0f1", anchor="w").pack(fill=tk.X)

        #composicione de nucleotido
        composition = self.get_sequence_composition()
        comp_text = f"Composición: A: {composition['A']}%, T: {composition['T']}%, G: {composition['G']}%, C: {composition['C']}%"
        tk.Label(info_frame, text=comp_text, font=("Helvetica", 12),
                 bg="#ecf0f1", anchor="w").pack(fill=tk.X)

        gc_content = composition['G'] + composition['C']
        tk.Label(info_frame, text=f"Contenido GC: {gc_content:.2f}%", font=("Helvetica", 12),
                 bg="#ecf0f1", anchor="w").pack(fill=tk.X)

        #grafico de composicion
        fig_frame = tk.Frame(seq_view_frame, bg="#ecf0f1")
        fig_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        #grafico de barras de composicion
        bases = ['A', 'T', 'G', 'C']
        values = [composition[base] for base in bases]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        ax1.bar(bases, values, color=colors)
        ax1.set_title('Composición de Nucleótidos')
        ax1.set_ylabel('Porcentaje (%)')

        #grafico de contenido GC por ventana
        window_size = 100
        gc_content_sliding = self.calculate_gc_sliding_window(window_size)
        positions = list(range(window_size // 2, len(self.sequence) - window_size // 2 + 1, window_size))

        ax2.plot(positions, gc_content_sliding, color='#2c3e50')
        ax2.set_title(f'Contenido GC (ventana de {window_size} bp)')
        ax2.set_xlabel('Posición en la secuencia')
        ax2.set_ylabel('Contenido GC (%)')

        plt.tight_layout()

        #incorporar los graficos en Tkinter
        canvas = FigureCanvasTkAgg(fig, fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        #area para mostrar la secuencia
        seq_frame = tk.Frame(seq_view_frame, bg="#ecf0f1")
        seq_frame.pack(fill=tk.BOTH, pady=10)

        tk.Label(seq_frame, text="Secuencia:", font=("Helvetica", 12, "bold"),
                 bg="#ecf0f1", anchor="w").pack(fill=tk.X, pady=(10, 5))

        #mostrar solo los primeros 100 nucleótidos para evitar lentitud
        preview_length = min(100, len(self.sequence))
        preview_text = str(self.sequence[:preview_length])
        if len(self.sequence) > preview_length:
            preview_text += "..."

        seq_text = scrolledtext.ScrolledText(seq_frame, wrap=tk.WORD, width=80, height=8,
                                             font=("Courier", 10))
        seq_text.pack(fill=tk.BOTH, expand=True)
        seq_text.insert(tk.END, preview_text)
        seq_text.config(state=tk.DISABLED)

    def get_sequence_composition(self):
        #calcular la composicion de nucleotidos
        seq_str = str(self.sequence).upper()
        total = len(seq_str)
        composition = {}

        for base in ['A', 'T', 'G', 'C']:
            count = seq_str.count(base)
            composition[base] = round(count / total * 100, 2)

        return composition

    def calculate_gc_sliding_window(self, window_size=100):
        #calcular el contenido GC en ventanas deslizantes
        seq_str = str(self.sequence).upper()
        gc_contents = []

        for i in range(0, len(seq_str) - window_size + 1, window_size):
            window = seq_str[i:i + window_size]
            gc_count = window.count('G') + window.count('C')
            gc_percent = (gc_count / window_size) * 100
            gc_contents.append(gc_percent)

        return gc_contents

    def find_orfs(self, min_length=300, genetic_code=1):
        #encotrar orfs
        orfs = []

        #buscar ORFs en los 6 marcos de lectura
        for strand, seq in [(+1, self.sequence), (-1, self.sequence.reverse_complement())]:
            strand_name = "directa" if strand == 1 else "inversa"

            #buscar en los 3 marcos de lectura
            for frame in range(3):
                frame_seq = seq[frame:]

                #encontrar todos los codones de inicio y fin
                start_positions = []
                for match in re.finditer('ATG', str(frame_seq), re.I):
                    if (match.start() % 3) == 0:
                        start_positions.append(match.start())

                for start in start_positions:
                    #buscar el primer codon de stop en marco dsp del inicio
                    for i in range(start, len(frame_seq), 3):
                        if i + 3 > len(frame_seq):
                            break

                        codon = frame_seq[i:i + 3]
                        if str(codon).upper() in ['TAA', 'TAG', 'TGA']:
                            #longitud del ORF = final - inicio
                            orf_length = i + 3 - start

                            if orf_length >= min_length:
                                orf_seq = frame_seq[start:i + 3]
                                protein = orf_seq.translate(table=genetic_code, to_stop=True)

                                #ajustar posiciones segun el marco y la cadena
                                if strand == 1:
                                    begin = frame + start
                                    end = frame + i + 3
                                else:
                                    #ajustar posiciones para la cadena inversa
                                    begin = len(self.sequence) - (frame + i + 3)
                                    end = len(self.sequence) - (frame + start)

                                #calcular puntuacion basica basada en longitud y patrones
                                score = self.calculate_orf_score(orf_seq, protein)

                                orfs.append({
                                    'strand': strand_name,
                                    'frame': frame,
                                    'start': begin + 1,  # +1 para coordenadas 1-based
                                    'end': end,
                                    'length': orf_length,
                                    'protein_length': len(protein),
                                    'sequence': str(orf_seq),
                                    'protein': str(protein),
                                    'score': score
                                })

                            break

        #ordenar ORFs por puntuación y longitud
        orfs.sort(key=lambda x: (x['score'], x['length']), reverse=True)
        return orfs

    def calculate_orf_score(self, orf_seq, protein_seq):
        #calculo simple de puntuación para un ORF
        score = 0

        #factor 1: Longitud del ORF (+ largo suele ser mejor)
        length_score = min(len(orf_seq) / 3000, 1.0) * 5
        score += length_score

        #factor 2: Contenido GC (valores cercanos a 50% suelen ser mejores)
        gc_content = (str(orf_seq).upper().count('G') + str(orf_seq).upper().count('C')) / len(orf_seq)
        gc_score = (1 - abs(gc_content - 0.5) * 2) * 3  # Puntuación max cuando GC = 50%
        score += gc_score

        #factor 3: Presencia de dominios conservados (simplificado)
        if 'KRRK' in str(protein_seq) or 'RRRK' in str(protein_seq):
            score += 1

        return round(score, 2)

    def show_orf_finder(self):
        if not self.sequence_loaded:
            messagebox.showwarning("Advertencia", "Primero debes cargar una secuencia.")
            return

        self.clear_main_content()

        #frame principal
        orf_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        orf_frame.pack(fill=tk.BOTH, expand=True)

        #titulo
        title = tk.Label(orf_frame, text="Predicción de Genes (ORFs)",
                         font=("Helvetica", 18, "bold"), bg="#ecf0f1")
        title.pack(pady=10)

        #panel de parametro
        param_frame = tk.Frame(orf_frame, bg="#ecf0f1", padx=10, pady=10)
        param_frame.pack(fill=tk.X)

        #longitud min del ORF
        tk.Label(param_frame, text="Longitud mínima (bp):", bg="#ecf0f1").grid(row=0, column=0, padx=5, pady=5)
        min_length_var = tk.StringVar(value="300")
        min_length_entry = tk.Entry(param_frame, textvariable=min_length_var, width=10)
        min_length_entry.grid(row=0, column=1, padx=5, pady=5)

        #codigo genético
        tk.Label(param_frame, text="Código genético:", bg="#ecf0f1").grid(row=0, column=2, padx=5, pady=5)
        genetic_code_var = tk.StringVar(value="1")
        genetic_code_combo = ttk.Combobox(param_frame, textvariable=genetic_code_var, width=25)
        genetic_code_combo['values'] = [
            "1: Estándar",
            "2: Mitocondrial de Vertebrados",
            "11: Bacteriano, Arqueobacteriano y Plastidial"
        ]
        genetic_code_combo.current(0)
        genetic_code_combo.grid(row=0, column=3, padx=5, pady=5)

        #boton para buscar ORFs
        search_btn = tk.Button(param_frame, text="Buscar ORFs",
                               font=("Helvetica", 10), bg="#2980b9", fg="white",
                               command=lambda: self.search_orfs(min_length_var, genetic_code_var, results_tree,
                                                                progress))
        search_btn.grid(row=0, column=4, padx=15, pady=5)

        #progress bar
        progress = ttk.Progressbar(orf_frame, orient="horizontal", length=300, mode="indeterminate")
        progress.pack(pady=10)

        #frame para botones de accion
        button_frame = tk.Frame(orf_frame, bg="#ecf0f1")
        button_frame.pack(fill=tk.X, pady=5)

        #boton para exportar resultados
        export_btn = tk.Button(button_frame, text="Exportar Resultados",
                               font=("Helvetica", 10), bg="#27ae60", fg="white",
                               command=lambda: self.export_orfs())
        export_btn.pack(side=tk.RIGHT, padx=10)

        #marco para resultados
        results_frame = tk.Frame(orf_frame, bg="#ecf0f1")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        #vista de arbol para resultados
        columns = ("ID", "Cadena", "Marco", "Inicio", "Fin", "Longitud (bp)", "Longitud (aa)", "Puntuación")
        results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)

        #config columnas
        results_tree.heading("ID", text="ID")
        results_tree.heading("Cadena", text="Cadena")
        results_tree.heading("Marco", text="Marco")
        results_tree.heading("Inicio", text="Inicio")
        results_tree.heading("Fin", text="Fin")
        results_tree.heading("Longitud (bp)", text="Longitud (bp)")
        results_tree.heading("Longitud (aa)", text="Longitud (aa)")
        results_tree.heading("Puntuación", text="Puntuación")

        #ajustar anchos de columna
        results_tree.column("ID", width=50)
        results_tree.column("Cadena", width=80)
        results_tree.column("Marco", width=60)
        results_tree.column("Inicio", width=100)
        results_tree.column("Fin", width=100)
        results_tree.column("Longitud (bp)", width=100)
        results_tree.column("Longitud (aa)", width=100)
        results_tree.column("Puntuación", width=100)

        #scorllbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_tree.yview)
        results_tree.configure(yscrollcommand=scrollbar.set)

        results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        #frame para detalles
        details_frame = tk.Frame(orf_frame, bg="#ecf0f1", padx=10, pady=10)
        details_frame.pack(fill=tk.BOTH, pady=10)

        tk.Label(details_frame, text="Detalles del ORF seleccionado:", font=("Helvetica", 12, "bold"),
                 bg="#ecf0f1").pack(anchor="w")

        details_text = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD, font=("Courier", 10))
        details_text.pack(fill=tk.BOTH, expand=True)
        details_text.insert(tk.END, "Selecciona un ORF para ver sus detalles.")
        details_text.config(state=tk.DISABLED)

        #vincular selección con visualización de detalles
        results_tree.bind("<<TreeviewSelect>>", lambda event: self.show_orf_details(event, details_text))

    def search_orfs(self, min_length_var, genetic_code_var, results_tree, progress):
        try:
            min_length = int(min_length_var.get())
            genetic_code = int(genetic_code_var.get().split(":")[0])

            #validar parametro
            if min_length < 0:
                messagebox.showerror("Error", "La longitud mínima debe ser un número positivo.")
                return

            #limpiar resultados anteriores
            for item in results_tree.get_children():
                results_tree.delete(item)

            #iniciar busqueda en un hilo separado
            progress.start()

            def search_thread():
                #encontrar ORFs
                self.predictions = self.find_orfs(min_length, genetic_code)

                #actualizar interfaz de usuario
                self.root.after(0, lambda: self.update_orf_results(results_tree, progress))

            threading.Thread(target=search_thread).start()
            self.current_results_tree = results_tree  #guardar la referencia para reusarla luego


        except ValueError:
            messagebox.showerror("Error", "Los parámetros deben ser valores numéricos.")
            progress.stop()

    def update_orf_results(self, results_tree, progress):
        #stop la barra de progreso
        progress.stop()

        #show resultados
        if not self.predictions:
            messagebox.showinfo("Resultados", "No se encontraron ORFs que cumplan con los criterios.")
            return

        #actualizar tabla de resultados
        for i, orf in enumerate(self.predictions):
            orf_id = f"ORF_{i + 1}"
            results_tree.insert("", "end", values=(
                orf_id,
                orf['strand'],
                orf['frame'],
                orf['start'],
                orf['end'],
                orf['length'],
                orf['protein_length'],
                orf['score']
            ))

        messagebox.showinfo("Resultados", f"Se encontraron {len(self.predictions)} ORFs potenciales.")

    def show_score_comparison(self):
        if not self.sequence_loaded or not self.predictions:
            messagebox.showwarning("Advertencia", "Primero debes cargar una secuencia y ejecutar la búsqueda de ORFs.")
            return

        self.clear_main_content()

        frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Comparación de Scores: Heurística vs ML",
                 font=("Helvetica", 16, "bold"), bg="#ecf0f1").pack(pady=10)

        #tabla de comparacion
        columns = ("ID", "Longitud", "Heurístico", "ML", "Diferencia")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER)

        tree.pack(fill=tk.BOTH, expand=True)

        for i, orf in enumerate(self.predictions):
            heur_score = self.calculate_orf_score(orf['sequence'], orf['protein'])
            ml_score = round(self.orf_scorer.score_orf(str(orf['sequence']), str(orf['protein'])), 2)
            orf['heuristic_score'] = heur_score
            orf['ml_score'] = ml_score
            diff = round(ml_score - heur_score, 2)

            tree.insert("", "end", values=(f"ORF_{i + 1}", orf['length'], heur_score, ml_score, diff))

        #colorear filas con contradiccion
        def colorize_rows():
            for i, row in enumerate(tree.get_children()):
                values = tree.item(row)['values']
                heur = float(values[2])
                ml = float(values[3])
                diff = abs(ml - heur)

                if diff > 3:
                    tree.tag_configure("conflict", background="#f9e79f")
                    tree.item(row, tags=("conflict",))

        colorize_rows()

        #details al seleccionar
        detail_box = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, font=("Courier", 10))
        detail_box.pack(fill=tk.BOTH, pady=10)
        detail_box.insert(tk.END, "Selecciona un ORF para ver detalles.")
        detail_box.config(state=tk.DISABLED)

        def show_detail(event):
            selected = tree.selection()
            if not selected:
                return
            idx = int(tree.item(selected[0])['values'][0].split('_')[1]) - 1
            orf = self.predictions[idx]

            detail_box.config(state=tk.NORMAL)
            detail_box.delete(1.0, tk.END)
            detail_box.insert(tk.END, f"ORF #{idx + 1}\n")
            detail_box.insert(tk.END, f"Inicio: {orf['start']} | Fin: {orf['end']}\n")
            detail_box.insert(tk.END, f"Score Heurístico: {orf['heuristic_score']}\n")
            detail_box.insert(tk.END, f"Score ML: {orf['ml_score']}\n")
            detail_box.insert(tk.END, f"Diferencia: {orf['ml_score'] - orf['heuristic_score']:.2f}\n")
            detail_box.insert(tk.END, f"Cadena: {orf['strand']} | Marco: {orf['frame']}\n\n")
            detail_box.insert(tk.END, f"Secuencia (primeros 100 nt):\n{orf['sequence'][:100]}")
            detail_box.config(state=tk.DISABLED)

        tree.bind("<<TreeviewSelect>>", show_detail)

    def show_orf_visualizer(self):
        if not self.sequence_loaded or not self.predictions:
            messagebox.showwarning("Advertencia", "Primero debes cargar una secuencia y ejecutar la búsqueda de ORFs.")
            return

        self.clear_main_content()

        frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Visualización de ORFs detectados",
                 font=("Helvetica", 16, "bold"), bg="#ecf0f1").pack(pady=10)

        nav_frame = tk.Frame(frame, bg="#ecf0f1")
        nav_frame.pack()

        chunk_size = 200
        self.seq_offset = 0

        text_area = tk.Text(frame, wrap=tk.WORD, font=("Courier", 10), width=100, height=15)
        text_area.pack(pady=10, expand=True)

        def color_orfs_in_chunk():
            text_area.config(state=tk.NORMAL)
            text_area.delete("1.0", tk.END)

            start = self.seq_offset
            end = min(self.seq_offset + chunk_size, len(self.sequence))
            chunk_seq = str(self.sequence[start:end])
            text_area.insert(tk.END, chunk_seq)

            #marcar ORFs que estan en este fragmento
            for i, orf in enumerate(self.predictions):
                if orf["start"] >= start + 1 and orf["start"] <= end:
                    s = orf["start"] - start - 1
                    e = orf["end"] - start
                    if 0 <= s < len(chunk_seq):
                        tag = f"orf_{i}"
                        text_area.tag_add(tag, f"1.{s}", f"1.{min(e, len(chunk_seq))}")
                        score = orf['score']
                        color = "#2ecc71" if score > 8 else "#f39c12" if score > 5 else "#e74c3c"
                        text_area.tag_config(tag, background=color, foreground="white")

            text_area.config(state=tk.DISABLED)

        def next_chunk():
            if self.seq_offset + chunk_size < len(self.sequence):
                self.seq_offset += chunk_size
                color_orfs_in_chunk()

        def prev_chunk():
            if self.seq_offset - chunk_size >= 0:
                self.seq_offset -= chunk_size
                color_orfs_in_chunk()

        tk.Button(nav_frame, text="⟵ Anterior", command=prev_chunk,
                  bg="#3498db", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Siguiente ⟶", command=next_chunk,
                  bg="#3498db", fg="white").pack(side=tk.LEFT, padx=5)

        #area de detalles al hacer clic
        detail_box = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, font=("Helvetica", 10))
        detail_box.pack(fill=tk.BOTH, pady=10)
        detail_box.insert(tk.END, "Haz clic sobre un ORF resaltado para ver detalles.")
        detail_box.config(state=tk.DISABLED)

        def show_orf_details_event(event):
            index = text_area.index(f"@{event.x},{event.y}")
            char_index = int(index.split('.')[1])
            genomic_pos = self.seq_offset + char_index + 1

            for i, orf in enumerate(self.predictions):
                if orf['start'] <= genomic_pos <= orf['end']:
                    detail_box.config(state=tk.NORMAL)
                    detail_box.delete(1.0, tk.END)
                    detail_box.insert(tk.END, f"ORF #{i + 1}\n")
                    detail_box.insert(tk.END, f"Inicio: {orf['start']} | Fin: {orf['end']} | Score: {orf['score']}\n")
                    detail_box.insert(tk.END, f"Longitud: {orf['length']} bp, {orf['protein_length']} aa\n")
                    detail_box.insert(tk.END, f"Cadena: {orf['strand']} | Marco: {orf['frame']}\n\n")
                    detail_box.insert(tk.END, "Primeros 100 nt:\n")
                    detail_box.insert(tk.END, orf['sequence'][:100] + ("..." if len(orf['sequence']) > 100 else ""))
                    detail_box.config(state=tk.DISABLED)
                    break

        text_area.bind("<Button-1>", show_orf_details_event)

        color_orfs_in_chunk()

    def show_orf_details(self, event, details_text):
        #mostrar detalles del ORF seleccionado
        selected_items = event.widget.selection()
        if not selected_items:
            return

        item = selected_items[0]
        idx = int(event.widget.item(item)['values'][0].split('_')[1]) - 1

        if idx < 0 or idx >= len(self.predictions):
            return

        orf = self.predictions[idx]

        #actualizar texto de detalles
        details_text.config(state=tk.NORMAL)
        details_text.delete(1.0, tk.END)

        details = f"ID: ORF_{idx + 1}\n"
        details += f"Posición: {orf['start']} - {orf['end']} ({orf['strand']})\n"
        details += f"Marco de lectura: {orf['frame']}\n"
        details += f"Longitud: {orf['length']} bp, {orf['protein_length']} aa\n"
        details += f"Puntuación: {orf['score']}\n\n"

        details += "Secuencia de nucleótidos:\n"
        seq = orf['sequence']
        #mostrar secuencia formateada en líneas de 60 caracteres
        for i in range(0, len(seq), 60):
            details += seq[i:i + 60] + "\n"

        details += "\nSecuencia de aminoácidos:\n"
        protein = orf['protein']
        for i in range(0, len(protein), 60):
            details += protein[i:i + 60] + "\n"

        details_text.insert(tk.END, details)
        details_text.config(state=tk.DISABLED)

    def export_orfs(self):
        if not self.predictions:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar.")
            return

        #solictar ubicación para guardar
        file_path = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".fasta",
            filetypes=[("Archivos FASTA", "*.fasta"), ("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                for i, orf in enumerate(self.predictions):
                    #formato FASTA para secuencia de nucleótidos
                    f.write(
                        f">ORF_{i + 1}|{self.seq_name}|{orf['start']}-{orf['end']}|{orf['strand']}|frame={orf['frame']}|score={orf['score']}\n")

                    #escribir secuencia en líneas de 60 caracteres
                    seq = orf['sequence']
                    for j in range(0, len(seq), 60):
                        f.write(seq[j:j + 60] + "\n")

                    #proteina traducida
                    f.write(
                        f">ORF_{i + 1}_protein|{self.seq_name}|{orf['start']}-{orf['end']}|length={orf['protein_length']}\n")
                    protein = orf['protein']
                    for j in range(0, len(protein), 60):
                        f.write(protein[j:j + 60] + "\n")

            messagebox.showinfo("Exportación", f"Los resultados han sido exportados a {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron exportar los resultados: {str(e)}")

    def show_validation(self):
        if not self.sequence_loaded:
            messagebox.showwarning("Advertencia", "Primero debes cargar una secuencia.")
            return

        if not self.predictions:
            messagebox.showwarning("Advertencia", "Primero debes ejecutar la predicción de genes.")
            return

        self.clear_main_content()

        #frame principal
        validate_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        validate_frame.pack(fill=tk.BOTH, expand=True)
        #titulo
        title = tk.Label(validate_frame, text="Validación de Predicciones",
                         font=("Helvetica", 18, "bold"), bg="#ecf0f1")
        title.pack(pady=10)

        # descripcion
        desc = tk.Label(validate_frame,
                        text="Valida tus predicciones de genes comparándolas con bases de datos externas.",
                        font=("Helvetica", 12), bg="#ecf0f1", wraplength=700)
        desc.pack(pady=10)

        # panel de seleccion
        select_frame = tk.Frame(validate_frame, bg="#ecf0f1")
        select_frame.pack(fill=tk.X, pady=10)

        tk.Label(select_frame, text="Selecciona un ORF para validar:",
                 bg="#ecf0f1").grid(row=0, column=0, padx=5, pady=5)

        #dropdown para seleccionar ORF
        orf_var = tk.StringVar()
        orf_dropdown = ttk.Combobox(select_frame, textvariable=orf_var, width=30)
        orf_dropdown['values'] = [f"ORF_{i + 1} ({orf['start']}-{orf['end']}, {orf['protein_length']} aa)"
                                  for i, orf in enumerate(self.predictions)]
        if len(self.predictions) > 0:
            orf_dropdown.current(0)
        orf_dropdown.grid(row=0, column=1, padx=5, pady=5)

        #dropdown para seleccionar base de datos
        tk.Label(select_frame, text="Base de datos:",
                 bg="#ecf0f1").grid(row=0, column=2, padx=5, pady=5)

        db_var = tk.StringVar(value="nr")
        db_dropdown = ttk.Combobox(select_frame, textvariable=db_var, width=20)
        db_dropdown['values'] = ["nr", "refseq_protein", "swissprot", "pdb"]
        db_dropdown.grid(row=0, column=3, padx=5, pady=5)

        #btn para realizar la validación
        validate_btn = tk.Button(select_frame, text="Validar",
                                 font=("Helvetica", 10), bg="#2980b9", fg="white",
                                 command=lambda: self.validate_orf(orf_var, db_var, results_text, progress))
        validate_btn.grid(row=0, column=4, padx=15, pady=5)

        #Progressbar
        progress = ttk.Progressbar(validate_frame, orient="horizontal", length=300, mode="indeterminate")
        progress.pack(pady=10)

        #area para mostrar resultados
        tk.Label(validate_frame, text="Resultados de la validación:",
                 font=("Helvetica", 12, "bold"), bg="#ecf0f1").pack(anchor="w", pady=(10, 5))

        results_text = scrolledtext.ScrolledText(validate_frame, height=20, wrap=tk.WORD)
        results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        results_text.insert(tk.END, "Selecciona un ORF y haz clic en 'Validar' para ver los resultados.")
        results_text.config(state=tk.DISABLED)

    def score_orfs(self):
        if not self.predictions:
            messagebox.showwarning("Advertencia", "Primero debes predecir los ORFs.")
            return

        for orf in self.predictions:
            orf['score'] = round(self.orf_scorer.score_orf(str(orf['sequence']), str(orf['protein'])), 2)

        #ordenar los ORFs por puntuacion
        self.predictions = sorted(self.predictions, key=lambda x: x['score'], reverse=True)

        #actualizar tabla si ya fue creada
        if hasattr(self, 'current_results_tree'):
            tree = self.current_results_tree

            #clean la tabla
            for item in tree.get_children():
                tree.delete(item)

            #insertar ORFs actualizados
            for i, orf in enumerate(self.predictions):
                orf_id = f"ORF_{i + 1}"
                tree.insert("", "end", values=(
                    orf_id,
                    orf['strand'],
                    orf['frame'],
                    orf['start'],
                    orf['end'],
                    orf['length'],
                    orf['protein_length'],
                    orf['score']
                ))

            messagebox.showinfo("Actualizado", "Las puntuaciones de los ORFs se han actualizado con ML.")
        else:
            messagebox.showinfo("Puntuado",
                                "Las puntuaciones se han actualizado, pero no se puede mostrar porque la tabla no ha sido inicializada.")

    def validate_orf(self, orf_var, db_var, results_text, progress):
        #obtener índice del ORF seleccionado
        if not orf_var.get():
            messagebox.showwarning("Advertencia", "Selecciona un ORF para validar.")
            return

        #extraer índice del ORF
        idx = int(orf_var.get().split('_')[1].split(' ')[0]) - 1
        if idx < 0 or idx >= len(self.predictions):
            messagebox.showerror("Error", "Índice de ORF no válido.")
            return

        #obtener base de datos seleccionada
        database = db_var.get()

        #iniciar busqueda en un hilo separado
        progress.start()

        #limpiar resultados anteriores
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, "Realizando búsqueda BLAST en NCBI. Este proceso puede tomar varios minutos...\n")
        results_text.config(state=tk.DISABLED)

        def search_thread():
            try:
                #realizar búsqueda BLAST
                orf = self.predictions[idx]
                protein_seq = orf['protein']

                #usar funcion de bussqueda BLAST 
                search_results = self.perform_blast_search(protein_seq, database)

                #actualizar UI
                self.root.after(0, lambda: self.update_validation_results(search_results, results_text, progress))

            except Exception as e:
                #manejar errs
                self.root.after(0, lambda: self.handle_validation_error(str(e), results_text, progress))

        threading.Thread(target=search_thread).start()

    def perform_blast_search(self, protein_seq, database):

        try:
            #config NCBI
            Entrez.email = "ja2958110@gmail.com" 
            Entrez.api_key = "6d1b113f5ac9c4441d6a2b43b0556a11a108"

            #parameters para la busequeda BLAST
            results = []

            #paso 1: enviar la consulta BLAST
            print("Enviando secuencia para análisis BLAST...")
            blast_handle = NCBIWWW.qblast(
                program="blastp",  # proteina contra proteína
                database=database,  # BD seleccionada
                sequence=protein_seq,  # seq de consulta
                expect=10,  # valor E max
                hitlist_size=10,  # num max de hits
                gapcosts="11 1",  # costos de gap
                matrix_name="BLOSUM62"  # matriz de sustitucion
            )

            #paso 2: analizar los resultados
            print("Analizando resultados BLAST...")
            blast_record = NCBIXML.read(blast_handle)

            #procesar cada alineamiento
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    #calcular porcentaje de identidad
                    identity_pct = (hsp.identities / hsp.align_length) * 100
                    #calcular cobertura
                    query_cover = (hsp.align_length / len(protein_seq)) * 100

                    #extraer info relevante
                    hit = {
                        'accession': alignment.accession,
                        'title': alignment.title,
                        'identity': identity_pct,
                        'e_value': hsp.expect,
                        'query_cover': query_cover,
                        'score': hsp.score,
                        'bits': hsp.bits
                    }
                    results.append(hit)

            return results

        except HTTPError as e:
            print(f"Error HTTP al conectar con NCBI: {e}")
            if "429" in str(e):
                raise Exception("Límite de solicitudes a NCBI excedido. Espera unos minutos e intenta de nuevo.")
            else:
                raise Exception(f"Error en el servidor NCBI: {str(e)}")
        except Exception as e:
            print(f"Error en la búsqueda BLAST: {e}")
            raise Exception(f"Error al realizar la búsqueda BLAST: {str(e)}")
        finally:
            if 'blast_handle' in locals():
                blast_handle.close()

    def update_validation_results(self, search_results, results_text, progress):
        #stop progressbar
        progress.stop()

        #actualizar resultados text
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)

        if not search_results:
            results_text.insert(tk.END, "No se encontraron coincidencias significativas en la base de datos NCBI.\n")
            results_text.insert(tk.END,
                                "Esto puede indicar un gen novedoso, una región no codificante, o parámetros de búsqueda muy restrictivos.")
            results_text.config(state=tk.DISABLED)
            return

        #show resultados
        results_text.insert(tk.END, f"Se encontraron {len(search_results)} coincidencias en NCBI BLAST:\n\n")

        #config etiquetas para formato
        results_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
        results_text.tag_configure("title", font=("Helvetica", 11, "bold"), foreground="#2980b9")
        results_text.tag_configure("good", foreground="#27ae60")
        results_text.tag_configure("moderate", foreground="#f39c12")
        results_text.tag_configure("poor", foreground="#e74c3c")
        results_text.tag_configure("link", foreground="#3498db", underline=1)

        for i, hit in enumerate(search_results):
            #titulo del hit con formato
            results_text.insert(tk.END, f"Hit #{i + 1}: ", "bold")
            results_text.insert(tk.END, f"{hit['title'][:80]}...\n", "title" if i == 0 else "")

            #info detallada
            results_text.insert(tk.END, f"Accesión: {hit['accession']}\n")

            #E-value con formato segun significancia
            results_text.insert(tk.END, "E-value: ")
            if hit['e_value'] < 1e-50:
                results_text.insert(tk.END, f"{hit['e_value']:.2e}\n", "good")
            elif hit['e_value'] < 1e-10:
                results_text.insert(tk.END, f"{hit['e_value']:.2e}\n", "moderate")
            else:
                results_text.insert(tk.END, f"{hit['e_value']:.2e}\n", "poor")

            #score
            results_text.insert(tk.END, f"Score: {hit['bits']:.1f} bits\n")

            #identidad con formato
            results_text.insert(tk.END, "Identidad: ")
            if hit['identity'] > 70:
                results_text.insert(tk.END, f"{hit['identity']:.2f}%\n", "good")
            elif hit['identity'] > 40:
                results_text.insert(tk.END, f"{hit['identity']:.2f}%\n", "moderate")
            else:
                results_text.insert(tk.END, f"{hit['identity']:.2f}%\n", "poor")

            #cobertura con formato
            results_text.insert(tk.END, "Cobertura: ")
            if hit['query_cover'] > 70:
                results_text.insert(tk.END, f"{hit['query_cover']:.2f}%\n", "good")
            elif hit['query_cover'] > 40:
                results_text.insert(tk.END, f"{hit['query_cover']:.2f}%\n", "moderate")
            else:
                results_text.insert(tk.END, f"{hit['query_cover']:.2f}%\n", "poor")

            #link a NCBI
            url = f"https://www.ncbi.nlm.nih.gov/protein/{hit['accession']}"
            results_text.insert(tk.END, f"Ver en NCBI: {url}\n\n", "link")

            #limitar a 5 hits para no sobrecargar UI
            if i >= 4:
                results_text.insert(tk.END, f"... y {len(search_results) - 5} resultados más\n\n")
                break

        #agregar interpretacion
        results_text.insert(tk.END, "Interpretación:\n", "bold")

        #determinar la calidad del mejor hit
        best_hit = search_results[0]
        if best_hit['identity'] > 80 and best_hit['query_cover'] > 80 and best_hit['e_value'] < 1e-50:
            results_text.insert(tk.END, "La predicción muestra alta similitud con proteínas conocidas. " +
                                "Alta probabilidad de que sea un gen funcional correctamente identificado.\n", "good")
        elif best_hit['identity'] > 50 and best_hit['query_cover'] > 50 and best_hit['e_value'] < 1e-10:
            results_text.insert(tk.END, "La predicción muestra similitud moderada con proteínas conocidas. " +
                                "Probablemente es un gen funcional, posiblemente con alguna divergencia evolutiva.\n",
                                "moderate")
        elif best_hit['identity'] > 30 and best_hit['query_cover'] > 30:
            results_text.insert(tk.END, "La predicción muestra baja similitud con proteínas conocidas. " +
                                "Podría ser un gen divergente, un pseudogen, o un dominio conservado dentro de una proteína mayor.\n",
                                "moderate")
        else:
            results_text.insert(tk.END, "La predicción muestra muy baja similitud con proteínas conocidas. " +
                                "Podría ser un gen novedoso, un falso positivo, o una región no codificante.\n", "poor")

        #Agregar nota sobre las limitaciones de la predicción
        results_text.insert(tk.END, "\nNota: La predicción de genes basada únicamente en ORFs tiene limitaciones, " +
                            "especialmente en organismos eucariotas donde los genes pueden contener intrones. " +
                            "Considera métodos adicionales para validar esta predicción.")

        results_text.config(state=tk.DISABLED)

    def handle_validation_error(self, error_msg, results_text, progress):
        #stop progress bar
        progress.stop()

        #show err msg
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, f"Error durante la búsqueda BLAST:\n{error_msg}\n\n")

        #proporcionar info adicional segun el tipo de error
        if "HTTPError" in error_msg or "conectar" in error_msg:
            results_text.insert(tk.END, "Posibles soluciones:\n")
            results_text.insert(tk.END, "1. Verifica tu conexión a internet\n")
            results_text.insert(tk.END, "2. Comprueba que tu clave API de NCBI sea válida\n")
            results_text.insert(tk.END, "3. Intenta más tarde, el servidor NCBI puede estar ocupado\n\n")

        if "límite" in error_msg.lower():
            results_text.insert(tk.END, "NCBI limita el número de consultas por usuario:\n")
            results_text.insert(tk.END, "- Con API key: 10 consultas/segundo, 3 solicitudes/segundo\n")
            results_text.insert(tk.END, "- Sin API key: 3 consultas/segundo, 1 solicitud/segundo\n\n")

        results_text.insert(tk.END,
                            "Información técnica: NCBI E-utilities requiere un tiempo de espera entre consultas. " +
                            "Si realizas múltiples búsquedas seguidas, es posible que debas esperar unos minutos.")

        results_text.config(state=tk.DISABLED)
        
    def find_orfs_in_sequence(self, seq, min_length=60, genetic_code=1):
        orfs = []
        for strand, s in [(+1, seq), (-1, seq.reverse_complement())]:
            strand_name = "directa" if strand == 1 else "inversa"
            for frame in range(3):
                frame_seq = s[frame:]
                for match in re.finditer('ATG', str(frame_seq), re.I):
                    if (match.start() % 3) == 0:
                        start = match.start()
                        for i in range(start, len(frame_seq), 3):
                            if i + 3 > len(frame_seq): break
                            codon = frame_seq[i:i+3]
                            if codon.upper() in ['TAA', 'TAG', 'TGA']:
                                length = i + 3 - start
                                if length >= min_length:
                                    orf_seq = frame_seq[start:i+3]
                                    protein = orf_seq.translate(table=genetic_code, to_stop=True)
                                    begin = frame + start if strand == 1 else len(seq) - (frame + i + 3)
                                    end = frame + i + 3 if strand == 1 else len(seq) - (frame + start)
                                    score = self.orf_scorer.score_orf(str(orf_seq), str(protein))
                                    orfs.append({
                                        'strand': strand_name,
                                        'frame': frame,
                                        'start': begin + 1,
                                        'end': end,
                                        'length': length,
                                        'protein_length': len(protein),
                                        'sequence': str(orf_seq),
                                        'protein': str(protein),
                                        'score': score,
                                        'is_smorf': length <= 300  # Marcar como smORF si es <= 300 nt
                                    })
                                break
        orfs.sort(key=lambda x: (x['score'], x['length']), reverse=True)
        return orfs

    
    def show_smorf_analysis(self):
        self.clear_main_content()
    
        self.smorf_predictions = []
    
        smorf_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=20, pady=20)
        smorf_frame.pack(fill=tk.BOTH, expand=True)
    
        tk.Label(smorf_frame, text="Análisis de smORF",
                 font=("Helvetica", 18, "bold"), bg="#ecf0f1").pack(pady=10)
    
        resultado_text = scrolledtext.ScrolledText(smorf_frame, height=10, wrap=tk.WORD,
                                                   font=("Courier", 10))
        resultado_text.pack(fill=tk.X, pady=10)
        resultado_text.insert(tk.END, "Carga una secuencia smORF para iniciar el análisis.")
        resultado_text.config(state=tk.DISABLED)
    
        #marco para tabla
        table_frame = tk.Frame(smorf_frame, bg="#ecf0f1")
        table_frame.pack(fill=tk.BOTH, expand=True)
    
        columns = ("ID", "Inicio", "Fin", "Marco", "Cadena", "Longitud", "AA", "Score")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
        #details
        details_text = scrolledtext.ScrolledText(smorf_frame, height=6, wrap=tk.WORD, font=("Courier", 10))
        details_text.pack(fill=tk.BOTH, pady=10)
        details_text.insert(tk.END, "Selecciona un ORF para ver detalles.")
        details_text.config(state=tk.DISABLED)
    
        def show_details(event):
            selected = tree.selection()
            if not selected:
                return
            idx = int(tree.item(selected[0])['values'][0].split('_')[1]) - 1
            orf = self.smorf_predictions[idx]
    
            details_text.config(state=tk.NORMAL)
            details_text.delete(1.0, tk.END)
            details_text.insert(tk.END, f"smORF #{idx + 1}\n")
            details_text.insert(tk.END, f"Inicio: {orf['start']} | Fin: {orf['end']} | Marco: {orf['frame']} | Cadena: {orf['strand']}\n")
            details_text.insert(tk.END, f"Longitud: {orf['length']} nt | {orf['protein_length']} aa\n")
            details_text.insert(tk.END, f"Score: {orf['score']}\n\n")
            details_text.insert(tk.END, f"Secuencia:\n{orf['sequence'][:100]}...\n")
            details_text.insert(tk.END, f"Aminoácidos:\n{orf['protein'][:100]}...\n")
            details_text.config(state=tk.DISABLED)
    
        tree.bind("<<TreeviewSelect>>", show_details)
    
        def cargar_smorf():
            file_path = filedialog.askopenfilename(
                title="Seleccionar archivo FASTA",
                filetypes=[("Archivos FASTA", "*.fasta *.fa"), ("Todos los archivos", "*.*")]
            )
            if not file_path:
                return
    
            try:
                record = list(SeqIO.parse(file_path, "fasta"))[0]
                smorf_seq = record.seq
                smorf_id = record.id
                aa_seq = smorf_seq.translate(to_stop=True)
                score = self.orf_scorer.score_orf(str(smorf_seq), str(aa_seq))
                gc = round((smorf_seq.count('G') + smorf_seq.count('C')) / len(smorf_seq) * 100, 2)
    
                resultado_text.config(state=tk.NORMAL)
                resultado_text.delete(1.0, tk.END)
                resultado_text.insert(tk.END, f"> {smorf_id}\n")
                resultado_text.insert(tk.END, f"Longitud: {len(smorf_seq)} nt\n")
                resultado_text.insert(tk.END, f"Contenido GC: {gc}%\n")
                resultado_text.insert(tk.END, f"Traducción (aa): {aa_seq}\n")
                resultado_text.insert(tk.END, f"Score ML: {round(score, 2)}\n")
                resultado_text.config(state=tk.DISABLED)
    
                #buscar ORFs en la smORF cargada
                self.smorf_predictions = self.find_orfs_in_sequence(smorf_seq, min_length=30)
    
                for item in tree.get_children():
                    tree.delete(item)
    
                for i, orf in enumerate(self.smorf_predictions):
                    orf_id = f"smORF_{i + 1}"
                    tree.insert("", "end", values=(
                        orf_id,
                        orf['start'],
                        orf['end'],
                        orf['frame'],
                        orf['strand'],
                        orf['length'],
                        orf['protein_length'],
                        orf['score']
                    ))
    
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo analizar el smORF: {str(e)}")
    
        # Botón de carga
        tk.Button(smorf_frame, text="Cargar smORF (.fasta)", font=("Helvetica", 11), bg="#27ae60",
                  fg="white", command=cargar_smorf).pack(pady=10)



    def show_about(self):
        self.clear_main_content()

        #frame principal
        about_frame = tk.Frame(self.main_content, bg="#ecf0f1", padx=30, pady=30)
        about_frame.pack(fill=tk.BOTH, expand=True)

        #titulo
        title = tk.Label(about_frame, text="Acerca de GeneFinder",
                             font=("Helvetica", 20, "bold"), bg="#ecf0f1")
        title.pack(pady=20)

        #descripcion del proyecto
        description = """
        GeneFinder es una herramienta educativa para la predicción de genes en secuencias genómicas
        desarrollada como proyecto de bioinformática.

        Características principales:
        • Carga de secuencias genómicas en formato FASTA
        • Análisis de composición de nucleótidos y contenido GC
        • Detección de marcos de lectura abiertos (ORFs) en las 6 posibles lecturas
        • Puntuación de potenciales genes basada en múltiples criterios
        • Validación de predicciones (simulada) contra bases de datos biológicas

        Esta herramienta implementa algoritmos básicos de predicción ab initio basados en:
        - Detección de codones de inicio (ATG) y terminación (TAA, TAG, TGA)
        - Evaluación del contenido GC y longitud de ORFs
        - Análisis de patrones de secuencia comunes en proteínas funcionales

        Fundamento científico:
        El análisis de marcos de lectura abiertos es uno de los métodos más básicos para
        la predicción de genes, especialmente en organismos procariotas donde los genes
        carecen de intrones. Este enfoque ha sido utilizado desde los inicios de la
        bioinformática y sigue siendo relevante hoy en día, aunque ahora se complementa
        con métodos más sofisticados como modelos ocultos de Markov y redes neuronales.
        """

        desc_text = scrolledtext.ScrolledText(about_frame, height=15, wrap=tk.WORD,
                                                  font=("Helvetica", 11), bg="white")
        desc_text.pack(fill=tk.BOTH, expand=True, pady=10)
        desc_text.insert(tk.END, description)
        desc_text.config(state=tk.DISABLED)

        #ref científicas
        ref_label = tk.Label(about_frame, text="Referencias científicas",
                                 font=("Helvetica", 14, "bold"), bg="#ecf0f1")
        ref_label.pack(anchor="w", pady=(20, 10))

        references = """
        1. Delcher, A. L., Bratke, K. A., Powers, E. C., & Salzberg, S. L. (2007). Identifying bacterial genes and endosymbiont DNA with Glimmer. Bioinformatics, 23(6), 673-679.

        2. Besemer, J., & Borodovsky, M. (2005). GeneMark: web software for gene finding in prokaryotes, eukaryotes and viruses. Nucleic acids research, 33(suppl_2), W451-W454.

        3. Salzberg, S. L., Delcher, A. L., Kasif, S., & White, O. (1998). Microbial gene identification using interpolated Markov models. Nucleic acids research, 26(2), 544-548.

        4. Hyatt, D., Chen, G. L., LoCascio, P. F., Land, M. L., Larimer, F. W., & Hauser, L. J. (2010). Prodigal: prokaryotic gene recognition and translation initiation site identification. BMC bioinformatics, 11(1), 119.
        """

        ref_text = scrolledtext.ScrolledText(about_frame, height=8, wrap=tk.WORD,
                                                 font=("Helvetica", 10), bg="white")
        ref_text.pack(fill=tk.BOTH, pady=10)
        ref_text.insert(tk.END, references)
        ref_text.config(state=tk.DISABLED)

        #btn para ver + recursos
        resources_btn = tk.Button(about_frame, text="Ver más recursos online",
                                    font=("Helvetica", 10), bg="#3498db", fg="white",
                                    command=lambda: webbrowser.open("https://www.ncbi.nlm.nih.gov/genbank/"))
        resources_btn.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = GenePredictionApp(root)
    root.mainloop()