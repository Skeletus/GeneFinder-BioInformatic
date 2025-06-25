# GeneFinder

## Descripción
GeneFinder es una herramienta educativa de escritorio para la **predicción de genes** (ORFs y smORFs) en secuencias genómicas. Integra algoritmos heurísticos clásicos y Machine Learning para facilitar la identificación, visualización y validación de regiones codificantes de ADN a través de una interfaz gráfica intuitiva.

## Características
* **Carga de archivos FASTA** con secuencias genómicas.
* **Detección de ORFs** en los 6 marcos de lectura posibles.
* **Puntuación de ORFs y smORFs** usando heurística y un modelo de Machine Learning (Random Forest).
* **Visualización interactiva** de la secuencia y de los ORFs detectados.
* **Comparación** de puntuaciones heurísticas vs. ML.
* **Validación automática** de genes predichos mediante BLAST (NCBI).
* **Exportación** de resultados en formato FASTA.
* **Interfaz gráfica amigable** usando Tkinter y gráficos con Matplotlib.

## Requisitos
- Python 3.6 o superior
- Bibliotecas:
  - tkinter (incluido con la mayoría de instalaciones de Python)
  - matplotlib
  - Biopython
  - numpy
  - requests

## Instalación
1. Clone este repositorio o descargue el código fuente
```bash
git clone https://github.com/Skeletus/GeneFinder-BioInformatic.git
cd GeneFinder-BioInformatic
```

2. Instale las dependencias
* biopython
* numpy
* matplotlib
* scikit-learn
* joblib
* requests

Instala las dependencias con:

```bash
pip install biopython numpy matplotlib scikit-learn joblib requests
```

3. Si tkinter no está disponible en su sistema:
   - Ubuntu/Debian: `sudo apt-get install python3-tk`
   - Fedora: `sudo dnf install python3-tkinter`
   - Windows: Reinstalar Python con la opción tk/tcl marcada
   - macOS: Suele venir incluido con Python

## Ejecución
```bash
python main.py
```

## Uso

1. Clona el repositorio o descarga los archivos.
2. Ejecuta la aplicación principal:

```bash
python main.py
```

3. Se abrirá la interfaz gráfica donde podrás:

   * Cargar un archivo FASTA
   * Detectar ORFs y smORFs
   * Visualizar y comparar puntuaciones
   * Validar predicciones usando BLAST
   * Exportar resultados
   * Consultar información y referencias en la sección "Acerca de"

---

## ¿Cómo funciona?

GeneFinder busca regiones candidatas a genes (ORFs) en las 6 lecturas posibles de la secuencia, calcula características biológicas (longitud, composición, contenido GC, presencia de motivos proteicos, fuerza de Kozak, etc.), y les asigna una puntuación heurística y una puntuación por ML. Puedes comparar ambos métodos y validar las predicciones usando BLAST.

El modelo de machine learning se puede reentrenar usando la clase `MLOrfScorer`, que permite crear datasets sintéticos o reales de genes y no-genes.

---

## Estructura del proyecto

```
GeneFinder/
│
├── main.py           # Interfaz gráfica y lógica principal
├── MLOrfScorer.py    # Puntuación de ORFs y funciones de Machine Learning
├── README.md         # Este archivo
└── ...
```

---

## Aplicaciones educativas
Esta herramienta es ideal para:
- Estudiantes de bioinformática que desean comprender los fundamentos de la predicción de genes
- Cursos introductorios de análisis genómico
- Proyectos de programación en biología computacional
- Demostración de conceptos básicos de anotación genómica

## Limitaciones
- No implementa algoritmos avanzados como modelos ocultos de Markov o redes neuronales
- La validación contra bases de datos es simulada con fines educativos
- Diseñado principalmente para secuencias procariotas (sin manejo de intrones)
- El rendimiento puede ser limitado para genomas grandes

## Autores

Proyecto desarrollado por:

* Arias Lopez, Jesus Arturo
* Fernandez Moreno, Nelly Belen
* Limachi Prieto, Mauro Benjamin
* Sandoval Arrieta, Mauricio Fabian

---
## Licencia
MIT License
