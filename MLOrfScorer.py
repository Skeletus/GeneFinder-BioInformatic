import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter


class MLOrfScorer:
    """
    Clase para puntuar ORFs utilizando tecnicas de Machine Learning
    """

    def __init__(self, model_path=None):
        #si se proporciona un modelo pre-entrenado, cargarlo
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.model_loaded = True
        else:
            #crear un modelo base
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model_loaded = False

        #especies comunes para entrenar/validar el modelo
        self.common_species = ['escherichia coli', 'bacillus subtilis', 'saccharomyces cerevisiae',
                               'homo sapiens', 'arabidopsis thaliana', 'drosophila melanogaster']

    def extract_features(self, orf_seq, protein_seq):
        """
        Extrae características relevantes del ORF y su proteína traducida
        """
        features = {}

        #convertir secuencias a string si son objetos Seq
        orf_seq = str(orf_seq).upper()
        protein_seq = str(protein_seq).upper()

        # 1. features básicas
        features['length'] = len(orf_seq)
        features['protein_length'] = len(protein_seq)

        # 2. composicion de nucleótidos
        for nuc in ['A', 'T', 'G', 'C']:
            features[f'freq_{nuc}'] = orf_seq.count(nuc) / len(orf_seq)

        # 3. contenido GC y GC en tercera pos del codon
        gc_count = orf_seq.count('G') + orf_seq.count('C')
        features['gc_content'] = (gc_count / len(orf_seq)) * 100 if len(orf_seq) > 0 else 0

        # GC en tercera pos del codon
        third_positions = orf_seq[2::3]
        gc_third = third_positions.count('G') + third_positions.count('C')
        features['gc_third'] = gc_third / len(third_positions) if third_positions else 0

        # 4. sesgo de uso de codones (Codon Usage Bias)
        codons = [orf_seq[i:i + 3] for i in range(0, len(orf_seq), 3) if i + 3 <= len(orf_seq)]
        codon_count = Counter(codons)
        total_codons = len(codons)

        # codones más frecuentes en genes reales
        common_codons = ['ATG', 'TGG', 'CAG', 'AAG', 'GAG', 'CTG']
        for codon in common_codons:
            features[f'codon_{codon}'] = codon_count.get(codon, 0) / total_codons if total_codons else 0

        # 5. properties de la proteína
        if len(protein_seq) > 0:
            try:
                # eliminar posibles caracteres no estandar
                clean_protein = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', protein_seq)
                if clean_protein:
                    prot_analysis = ProteinAnalysis(clean_protein)
                    features['protein_instability'] = prot_analysis.instability_index()
                    features['protein_aromaticity'] = prot_analysis.aromaticity()
                    features['protein_isoelectric'] = prot_analysis.isoelectric_point()

                    # contenido de aminoácidos importantes
                    aa_content = prot_analysis.get_amino_acids_percent()
                    hydrophobic = sum(aa_content.get(aa, 0) for aa in 'AVILMFYW')
                    features['hydrophobic_content'] = hydrophobic
                else:
                    # valores por defecto si la proteína no contiene aminoácidos estándar
                    features['protein_instability'] = 50  # neutral
                    features['protein_aromaticity'] = 0.1  # vaalor bajo por defecto
                    features['protein_isoelectric'] = 7.0  # neutral
                    features['hydrophobic_content'] = 0.5  # valor medio
            except Exception:
                # si hay errores en el analisi usar valores por defecto
                features['protein_instability'] = 50
                features['protein_aromaticity'] = 0.1
                features['protein_isoelectric'] = 7.0
                features['hydrophobic_content'] = 0.5
        else:
            features['protein_instability'] = 50
            features['protein_aromaticity'] = 0.1
            features['protein_isoelectric'] = 7.0
            features['hydrophobic_content'] = 0.5

        # 6. motivos funcionales y señales
        # signal de inicio fuerte (contexto Kozak)
        if len(orf_seq) >= 9:
            start_context = orf_seq[:9]
            #una secuencia Kozak fuerte generalmente tiene G en posición +4 y A/G en -3
            kozak_strength = 0
            if orf_seq[3:6] == 'ATG':  # asegurar de que el coodon de inicio es ATG
                if start_context[0] in 'AG':  # pos -3
                    kozak_strength += 0.5
                if start_context[6] == 'G':  # pos +4
                    kozak_strength += 0.5
            features['kozak_strength'] = kozak_strength
        else:
            features['kozak_strength'] = 0

        #motivoms comunes en proteínas
        protein_motifs = {
            'dna_binding': ['KRRK', 'RRRK', 'KKRK'],  # motivo de unión a DNA
            'nuclear_loc': ['PKKKRKV'],  # signal de localización nuclear
            'zinc_finger': ['CXXC'],  # motivo de dedos de zinc (simplificado)
            'phosphorylation': ['RXS', 'RXT', 'SXR', 'TXR']  # sitios de fosforilación
        }

        for motif_name, patterns in protein_motifs.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern.replace('X', '.'), protein_seq))
            features[f'motif_{motif_name}'] = count / len(protein_seq) if len(protein_seq) > 0 else 0

        # 7. indice de adaptación de codones (CAI)
        # nota es version simplifcada, ideal seria calcular
        #  con data especiifca  del organismo
        high_cai_codons = {'ATG', 'TGG', 'AAA', 'AAG', 'GAA', 'GAG', 'CAG', 'TAC', 'TTC', 'ATC', 'ACC', 'GCC', 'CTG'}
        cai_sum = sum(1 for codon in codons if codon in high_cai_codons)
        features['cai_estimate'] = cai_sum / total_codons if total_codons else 0

        return features

    def features_to_vector(self, features):
        """
        Convierte el diccionario de características en un vector
        """
        #definir el orden de las features para que sea consistente
        feature_names = [
            'length', 'protein_length',
            'freq_A', 'freq_T', 'freq_G', 'freq_C',
            'gc_content', 'gc_third',
            'codon_ATG', 'codon_TGG', 'codon_CAG', 'codon_AAG', 'codon_GAG', 'codon_CTG',
            'protein_instability', 'protein_aromaticity', 'protein_isoelectric', 'hydrophobic_content',
            'kozak_strength',
            'motif_dna_binding', 'motif_nuclear_loc', 'motif_zinc_finger', 'motif_phosphorylation',
            'cai_estimate'
        ]

        return [features.get(name, 0) for name in feature_names]

    def train_model(self, positive_examples, negative_examples):
        """
        Entrena el modelo con ejemplos positivos y negativos

        positive_examples: lista de tuplas (orf_seq, protein_seq) de genes reales
        negative_examples: lista de tuplas (orf_seq, protein_seq) de no-genes
        """
        X = []
        y = []

        #procesar ejemplos positivos
        for orf_seq, protein_seq in positive_examples:
            features = self.extract_features(orf_seq, protein_seq)
            X.append(self.features_to_vector(features))
            y.append(1)  # Clase positiva

        #procesar ejemplos negativos
        for orf_seq, protein_seq in negative_examples:
            features = self.extract_features(orf_seq, protein_seq)
            X.append(self.features_to_vector(features))
            y.append(0)  # Clase negativa

        #entrenar el modelo
        self.model.fit(np.array(X), np.array(y))
        self.model_loaded = True

    def score_orf(self, orf_seq, protein_seq):
        """
        Puntúa un ORF utilizando el modelo de ML o una heurística avanzada si no hay modelo
        """
        #extrae features
        features = self.extract_features(orf_seq, protein_seq)
        feature_vector = self.features_to_vector(features)

        if self.model_loaded:
            #utiliza el modelo para predecir la prob de que sea un gen real
            probability = self.model.predict_proba([feature_vector])[0][1]
            #normaliza a una puntuación entre 0 y 10
            score = probability * 10
        else:
            #si no hay modelo, usar una heuritisca más avanzada que la original
            score = self._heuristic_score(features)

        return round(score, 2)

    def _heuristic_score(self, features):
        """
        Puntuación heurística avanzada para usar cuando no hay modelo entrenado
        """
        score = 0

        # 1. longitud (+ largo suele ser mejor, pero hay un limit)
        length_score = min(features['length'] / 1500, 1.5)  # saturar en 1.5 para 1500 bp
        score += length_score

        # 2. contenido GC (premiar valores típicos segun el tipo de organismo)
        gc = features['gc_content']
        # valores típicos: bacterias ~50%, levaduras ~40%, mamíferos ~45%
        gc_score = 2.0 * (1 - min([abs(gc - 50), abs(gc - 40), abs(gc - 45)]) / 30)
        score += gc_score

        # 3. sesgo de uso de codones
        cai_score = features['cai_estimate'] * 2
        score += cai_score

        # 4. GC en tercera pos del codón (a menudo + alto en genes reales)
        gc3_score = features['gc_third'] * 1.5
        score += gc3_score

        # 5. presencia de motivos proteicos
        motif_score = (
                features['motif_dna_binding'] +
                features['motif_nuclear_loc'] * 2 +
                features['motif_zinc_finger'] * 1.5 +
                features['motif_phosphorylation']
        )
        score += min(motif_score, 2)  # limpiar a 2 puntos

        # 6. fuerza de la señal Kozak (inicio de traduccion)
        score += features['kozak_strength'] * 1.5

        # normalizar a una escala de 0-10
        score = min(score, 10)

        return score

    def save_model(self, model_path):
        """
        Guarda el modelo entrenado
        """
        if self.model_loaded:
            joblib.dump(self.model, model_path)
            return True
        return False

    def generate_synthetic_training_data(self):
        """
        Genera datos sintéticos para entrenar el modelo cuando no hay datos reales disponibles
        """
        positive_examples = []
        negative_examples = []

        #parameters para genes sintéticos
        gene_lengths = [600, 900, 1200, 1500, 2100]
        gc_contents = [0.35, 0.45, 0.55, 0.65]

        #generar ejemplos positivos (genes simulados)
        for length in gene_lengths:
            for gc in gc_contents:
                #generar un gen sintetico con distribución GC especifica
                orf_seq = self._generate_synthetic_gene(length, gc)
                protein_seq = self._translate(orf_seq)
                positive_examples.append((orf_seq, protein_seq))

        #generar ejemplos negativos (secuencias aleatorias)
        for length in [300, 450, 600, 900]:
            for gc in gc_contents:
                #secuencias completamente aleatorias
                random_seq = self._generate_random_sequence(length, gc)
                protein_seq = self._translate(random_seq)
                negative_examples.append((random_seq, protein_seq))

                #secuencias con codon de parada prematuro
                seq_with_stop = self._insert_early_stop_codon(self._generate_synthetic_gene(length, gc))
                protein_seq = self._translate(seq_with_stop)
                negative_examples.append((seq_with_stop, protein_seq))

        return positive_examples, negative_examples

    def _generate_synthetic_gene(self, length, gc_content):
        """
        Genera un gen sintético con características de genes reales
        """
        #asegurar que empieza con ATG
        gene = "ATG"

        #generar el resto del gen
        nucleotides = ['G', 'C'] * int(gc_content * 100) + ['A', 'T'] * int((1 - gc_content) * 100)
        np.random.shuffle(nucleotides)

        #evitar codones de parada internos
        i = 3
        while i < length - 3:
            codon = ''.join(np.random.choice(nucleotides, 3))
            if codon not in ['TAA', 'TAG', 'TGA']:
                gene += codon
                i += 3

        #agregar codon de parada
        gene += np.random.choice(['TAA', 'TAG', 'TGA'])

        return gene

    def _generate_random_sequence(self, length, gc_content):
        """
        Genera una secuencia aleatoria con un contenido GC específico
        """
        nucleotides = ['G', 'C'] * int(gc_content * 100) + ['A', 'T'] * int((1 - gc_content) * 100)
        np.random.shuffle(nucleotides)
        sequence = ''.join(np.random.choice(nucleotides, length))
        return sequence

    def _insert_early_stop_codon(self, sequence):
        """
        Inserta un codón de parada temprano en la secuencia
        """
        stop_codons = ['TAA', 'TAG', 'TGA']
        position = np.random.randint(3, len(sequence) // 3) * 3

        return sequence[:position] + np.random.choice(stop_codons) + sequence[position + 3:]

    def _translate(self, dna_seq):
        """
        Traduce una secuencia de DNA a proteína
        """
        genetic_code = {
            'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
            'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
            'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
            'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
            'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
            'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
            'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
            'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
            'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
            'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
        }

        protein = ""
        for i in range(0, len(dna_seq), 3):
            if i + 3 <= len(dna_seq):
                codon = dna_seq[i:i + 3]
                if codon in genetic_code:
                    aa = genetic_code[codon]
                    if aa == '_':  #stop codon
                        break
                    protein += aa

        return protein