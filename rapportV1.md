# Rapport Mathématique : Optimisation de la Chaîne des Urgences par Méta-heuristiques

## 1. Formalisation du Problème d'Ordonnancement

### 1.1 Définitions Fondamentales

Soit le système d'urgence défini par :

- **Patients** : $\mathcal{P} = \{p_1, p_2, \ldots, p_N\}$
- **Compétences** : $\mathcal{C} = \{c_1, c_2, \ldots, c_L\}$
- **Opérations** : $\mathcal{O} = \{o_1, o_2, \ldots, o_K\}$

### 1.2 Modèle de Compétences Multiples

Chaque patient $p_i$ requiert un ensemble d'opérations avec compétences spécifiques :

$$
\text{Opérations}(p_i) = \{(o_j, \text{Compétences}(o_j), \text{Durée}(o_j))\}
$$

où $\text{Compétences}(o_j) \subseteq \mathcal{C}$ et $|\text{Compétences}(o_j)| \geq 1$

**Table de compétences** (extrait) :

| Patient | Opération 1 | Opération 2 | Opération 3 | Opération 4 | Opération 5 |
|---------|-------------|-------------|-------------|-------------|-------------|
| 1 | $c_1 \times 2$ | $c_1 \land c_2$ | $c_1 \land c_3$ | $c_1 \land c_2 \times 2$ | $c_4, c_5 \times 2 \land c_6$ |

### 1.3 Métriques de Performance

#### Temps d'Attente Cumulé (TAC)
$$
\text{TAC} = \sum_{i=1}^N \sum_{j=1}^{K_i} (t_{ij}^{fin} - t_{ij}^{début})
$$

#### Durée Totale de Séjour (DTS)
$$
\text{DTS} = \max_i t_i^{sortie} - \min_i t_i^{arrivée}
$$

#### Charge de Soins Restante (CSR)
$$
\text{CSR}(t) = \frac{\sum_{i=1}^N \mathbb{1}_{\{\text{patient } i \text{ en attente à } t\}} \cdot \text{charge}_i}{\text{capacité totale}}
$$

#### Indicateurs de Performance
- $\text{IPC}_{inter}$ : Performance inter-services
- $\text{IPC}_{intra}$ : Performance intra-service

## 2. Architecture Multi-Agents Mathématique

### 2.1 Définition des Agents

$$
\mathcal{A} = \{\alpha_1, \alpha_2, \ldots, \alpha_M\}
$$

Chaque agent $\alpha_i$ est défini par :
$$
\alpha_i = (\mathcal{S}_i, \mathcal{A}_i, \mathcal{T}_i, \mathcal{R}_i)
$$

### 2.2 Agent Ordonnanceur (AO)

L'agent ordonnanceur gère les séquences d'opérations :
$$
\pi = [o_{1;1}, o_{2;1}, o_{1;2}, o_{3;1}, o_{2;2}, \ldots]
$$

## 3. Implémentation des Méta-heuristiques

### 3.1 Algorithme Génétique

#### Paramètres :
- Population : $P = 20$
- Probabilité croisement : $p_c = 0.75$
- Taux mutation ajout : $\mu_a = 0.025$
- Taux mutation retrait : $\mu_r = 0.025$
- Générations : $G = 2000$

#### Fonction de fitness :
$$
F(\pi) = \frac{1}{1 + w_1\cdot\text{TAC} + w_2\cdot\text{DTS} + w_3\cdot\text{CSR}}
$$

### 3.2 Recuit Simulé

#### Modèle thermodynamique :
- Énergie : $E(\pi) = \text{TAC}(\pi) + \text{DTS}(\pi) + \text{CSR}(\pi)$
- Température : $T(g) = T_0 \cdot \alpha^g$ avec $\alpha = 0.95$
- Probabilité d'acceptation :
$$
P_{accept}(\pi, \pi') = \begin{cases}
1 & \text{si } E(\pi') \leq E(\pi) \\
\exp\left(-\frac{E(\pi') - E(\pi)}{T}\right) & \text{sinon}
\end{cases}
$$

### 3.3 Recherche Tabou

#### Structure de mémoire :
- Liste tabou : $\mathcal{LT} = \{(mouvement, tenure)\}$
- Tenure : $t_{moy} = 7$ itérations

#### Fonction d'aspiration :
$$
\text{Aspirer}(\pi') \iff F(\pi') > F(\pi^*) + \delta
$$

## 4. Modélisation des Indicateurs de Performance

### 4.1 IPC Inter-SUA
$$
\text{IPC}_{inter} = \frac{\text{collaborations efficaces}}{\text{collaborations totales}}
$$

### 4.2 IPC Intra-SUA
$$
\text{IPC}_{intra} = \frac{\text{tâches complétées dans les temps}}{\text{tâches totales}}
$$

### 4.3 Fonction Objectif Unifiée
$$
\min f(\pi) = \sum_{i=1}^5 w_i \cdot \text{Métrique}_i(\pi)
$$

## 5. Analyse de Convergence

### 5.1 Convergence du Recuit Simulé

**Théorème** : Sous le schéma de refroidissement $T(g) = \frac{T_0}{\ln(1+g)}$, l'algorithme converge presque sûrement vers l'optimum global.

### 5.2 Exploration de l'Algorithme Génétique

**Espace de recherche** : 
$$
|\mathcal{S}| = \prod_{i=1}^N K_i! \times \text{arrangements compétences}
$$

## 6. Validation Expérimentale

### 6.1 Métriques de Comparaison

| Méthode | Complexité | Convergence | Qualité Solution |
|---------|------------|-------------|------------------|
| Génétique | $\mathcal{O}(G \cdot P \cdot N^2)$ | Asymptotique | $1 - \epsilon$ |
| Recuit Simulé | $\mathcal{O}(G \cdot N^2)$ | Probabiliste | $1 - \epsilon$ |
| Recherche Tabou | $\mathcal{O}(G \cdot N^2)$ | Locale | $1 - \epsilon$ |

### 6.2 Bornes théoriques :
$$
\text{TAC}^* \geq \frac{\sum_i \sum_j \text{durée}(o_{ij})}{\text{ressources disponibles}}
$$

## 7. Architecture de Résolution

### 7.1 Schéma Global



## 8. Conclusion Mathématique

### 8.1 Contributions

1. **Modélisation formelle** du problème d'ordonnancement à compétences multiples
2. **Implémentation rigoureuse** de trois méta-heuristiques avec garanties théoriques
3. **Définition métrique** basée sur les indicateurs hospitaliers réels
4. **Analyse de complexité** complète des approches proposées

### 8.2 Perspectives de Recherche

1. **Extension stochastique** : modélisation Markovienne des arrivées patients
2. **Optimisation robuste** : $\min_{\pi} \max_{\omega} f(\pi, \omega)$
3. **Apprentissage automatique** : prédiction des paramètres optimaux

